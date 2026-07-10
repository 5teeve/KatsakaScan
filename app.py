# app.py
import pickle
import numpy as np
import streamlit as st
from PIL import Image

from config import PATHS
from ml.scaler import MinMaxScaler
from features.extractor import extract_features

st.set_page_config(page_title="Diagnostic Rouille du Maïs", page_icon=None)


@st.cache_resource
def load_model():
    with open(PATHS.model_fs_pkl, "rb") as f:
        model = pickle.load(f)
    scaler = MinMaxScaler.load(PATHS.scaler_pkl)
    return model, scaler


model, scaler = load_model()
PATHS.uploads_dir.mkdir(exist_ok=True)

st.title("Diagnostic de la Rouille")

# --- Module 1 : Upload et prédiction en temps réel ---
uploaded_file = st.file_uploader("Téléverser une photo de feuille de maïs",
                                  type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image téléversée", width="stretch")

    save_path = PATHS.uploads_dir / uploaded_file.name
    image.save(save_path)

    fv = extract_features(save_path)
    X = scaler.transform(np.array([[fv.pct_rouille, fv.rugosite, fv.pct_jaunissement]]))
    pred = int(model.predict(X)[0])

    if pred == 1:
        st.error("ATTENTION : Feuille Malade (Rouille Detected)")
    else:
        st.success("Feuille Saine")

    st.caption(
        f"pct_rouille={fv.pct_rouille:.4f} · "
        f"rugosite={fv.rugosite:.2f} · "
        f"pct_jaunissement={fv.pct_jaunissement:.4f}"
    )

    diag_path = save_path.with_suffix(".txt")
    diag_path.write_text("Malade" if pred == 1 else "Saine")

# --- Module 2 : Galerie d'historique des détections ---
st.header("Historique des détections")

images = sorted(
    list(PATHS.uploads_dir.glob("*.jpg")) +
    list(PATHS.uploads_dir.glob("*.jpeg")) +
    list(PATHS.uploads_dir.glob("*.png")),
    key=lambda p: p.stat().st_mtime
)

if not images:
    st.info("Aucune image analysée pour le moment.")
else:
    cols = st.columns(4)
    for i, img_path in enumerate(images):
        diag_path = img_path.with_suffix(".txt")
        diagnostic = diag_path.read_text() if diag_path.exists() else "?"
        with cols[i % 4]:
            st.image(Image.open(img_path), width=200)
            st.caption(diagnostic)
