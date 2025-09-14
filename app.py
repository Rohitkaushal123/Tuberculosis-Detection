import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""     # no GPU on Streamlit Cloud
os.environ["OPENCV_FORCE_HEADLESS"] = "1"   # prevents libGL.so.1 crash

import streamlit as st
import numpy as np
import pydicom
from PIL import Image
import tempfile
import pandas as pd

# Import after setting env vars
import cv2
import torch
from ultralytics import YOLO

cv2.setNumThreads(1)
torch.set_num_threads(1)

# ======== settings (tweak if needed) ========
CONF_THR = 0.30          # lower => more sensitive (more positives)
INFER_MAX_SIDE = 640     # long side resized to this for inference
SICK_LABEL = "Pneumonia" # label text to show when sick
# ===========================================

st.set_page_config(page_title="Pneumonia Checker", layout="centered")
st.title("🩻 Pneumonia Checker (Simple)")
st.caption("Upload a chest X-ray (DICOM / PNG / JPG). If pneumonia is found, boxes will be shown.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
task = getattr(model, "task", None) or getattr(model.model, "task", None)

def dcm_to_rgb(dcm_path: str) -> np.ndarray:
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array.astype(np.float32)

    slope = float(getattr(dcm, "RescaleSlope", 1.0))
    intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
    img = img * slope + intercept

    wc = getattr(dcm, "WindowCenter", None)
    ww = getattr(dcm, "WindowWidth", None)
    if isinstance(wc, pydicom.multival.MultiValue):
        wc = float(wc[0])
    if isinstance(ww, pydicom.multival.MultiValue):
        ww = float(ww[0])

    if wc is not None and ww is not None and ww > 0:
        lo, hi = wc - ww/2, wc + ww/2
        img = np.clip(img, lo, hi)
    else:
        lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.5)
        img = np.clip(img, lo, hi)

    img = (img - img.min()) / max(1e-6, (img.max() - img.min()))
    img = (img * 255.0).astype(np.uint8)

    if str(getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")).upper() == "MONOCHROME1":
        img = 255 - img

    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def shrink_with_scale(img: np.ndarray, max_side: int = INFER_MAX_SIDE):
    """Return (resized_img, scale). scale = resized/original."""
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img, 1.0
    scale = max_side / float(m)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return small, scale

def scale_boxes_to_original(df: pd.DataFrame, scale: float) -> pd.DataFrame:
    if df.empty: 
        return df
    inv = 1.0 / max(scale, 1e-6)
    df = df.copy()
    for k in ["x1","y1","x2","y2"]:
        df[k] = df[k] * inv
    return df

def draw_boxes(img: np.ndarray, boxes_df: pd.DataFrame) -> np.ndarray:
    out = img.copy()
    for _, row in boxes_df.iterrows():
        x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{SICK_LABEL} {row["conf"]:.2f}'
        cv2.putText(out, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    return out

def boxes_df_from_result(r) -> pd.DataFrame:
    if r.boxes is None or len(r.boxes) == 0:
        return pd.DataFrame(columns=["conf","x1","y1","x2","y2"])
    b = r.boxes
    conf = b.conf.cpu().numpy()
    xyxy = b.xyxy.cpu().numpy()
    rows = []
    for i in range(len(conf)):
        rows.append({
            "conf": float(conf[i]),
            "x1": float(xyxy[i][0]),
            "y1": float(xyxy[i][1]),
            "x2": float(xyxy[i][2]),
            "y2": float(xyxy[i][3]),
        })
    return pd.DataFrame(rows)

uploaded = st.file_uploader("Upload X-ray", type=["dcm", "png", "jpg", "jpeg"])

if uploaded:
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(uploaded.read()); tmp.flush()
    path = tmp.name

    # read image
    if uploaded.name.lower().endswith(".dcm"):
        rgb_full = dcm_to_rgb(path)
    else:
        rgb_full = np.array(Image.open(path).convert("RGB"))

    st.image(rgb_full, caption="Input X-ray", use_container_width=True)

    # === PREDICT ===
    rgb_small, scale = shrink_with_scale(rgb_full, INFER_MAX_SIDE)
    results = model.predict(source=rgb_small, conf=CONF_THR, verbose=False, device="cpu")

    if task == "detect":
        # gather all boxes from results (usually 1 image -> 1 result)
        all_boxes_small = []
        for r in results:
            df = boxes_df_from_result(r)
            if not df.empty:
                all_boxes_small.append(df)
        if len(all_boxes_small) == 0:
            st.success("✅ NOT SICK")
        else:
            # rescale boxes back to original size and draw
            boxes_full = scale_boxes_to_original(pd.concat(all_boxes_small, ignore_index=True), scale)
            out = draw_boxes(rgb_full, boxes_full)
            st.error(f"⚠️ SICK ({SICK_LABEL})")
            st.image(out, caption="Detected region(s)", use_container_width=True)

    elif task == "classify":
        # classification: use top probability to decide SICK/NOT SICK (no boxes available)
        r = results[0]
        if r.probs is None:
            st.warning("Model returned no probabilities. Ensure it's a classification or detection YOLO model.")
        else:
            scores = r.probs.data.cpu().numpy().reshape(-1)
            top_id = int(scores.argmax())
            top_score = float(scores[top_id])
            names = r.names if hasattr(r, "names") else model.names
            top_name = names.get(top_id, str(top_id))
            is_pneumonia = ("pneumonia" in top_name.lower()) or (len(names) == 1)
            if is_pneumonia and top_score >= CONF_THR:
                st.error(f"⚠️ SICK ({SICK_LABEL}) — score {top_score:.2f}")
            else:
                st.success("✅ NOT SICK")
            st.caption("Note: classification models cannot show location/boxes. Train a detection model to localize.")
    else:
        st.warning("Unsupported model type. Use a YOLO 'detect' (boxes) or 'classify' (probabilities) model.")

