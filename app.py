# app.py
import pickle
import numpy as np
import streamlit as st
from PIL import Image

from config import PATHS
from ml.scaler import MinMaxScaler
from features.extractor import extract_features

st.set_page_config(page_title="Diagnostic Rouille du Mais", page_icon=None)


@st.cache_resource
def load_model():
    with open(PATHS.model_fs_pkl, "rb") as f:
        model = pickle.load(f)
    scaler = MinMaxScaler.load(PATHS.scaler_pkl)
    return model, scaler


model, scaler = load_model()
PATHS.uploads_dir.mkdir(exist_ok=True)

st.title("Diagnostic de la Rouille Polysora")

# --- Module 1 : Upload et prediction sur demande ---
uploaded_file = st.file_uploader("Televerser une photo de feuille de mais",
                                  type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image televersee", width="stretch")

    if st.button("Analyser la feuille"):
        with st.spinner("Analyse en cours..."):
            save_path = PATHS.uploads_dir / uploaded_file.name
            image.save(save_path)

            fv = extract_features(save_path)
            X = scaler.transform(np.array([[fv.pct_rouille, fv.rugosite, fv.pct_jaunissement]]))
            pred = int(model.predict(X)[0])

            diag_path = save_path.with_suffix(".txt")
            diag_path.write_text("Malade" if pred == 1 else "Saine")

        if pred == 1:
            st.error("Feuille Malade (Rouille detectee)")
        else:
            st.success("Feuille Saine")

        st.caption(
            f"pct_rouille={fv.pct_rouille:.4f} - "
            f"rugosite={fv.rugosite:.2f} - "
            f"pct_jaunissement={fv.pct_jaunissement:.4f}"
        )

# --- Module 2 : Galerie d'historique des detections ---
st.header("Historique des detections")

images = sorted(
    path for path in PATHS.uploads_dir.iterdir()
    if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

if not images:
    st.info("Aucune image analysee pour le moment.")
else:
    cols = st.columns(4)
    for i, img_path in enumerate(images):
        diag_path = img_path.with_suffix(".txt")
        diagnostic = diag_path.read_text() if diag_path.exists() else "?"
        with cols[i % 4]:
            st.image(str(img_path), width="stretch")
            st.caption(diagnostic)
