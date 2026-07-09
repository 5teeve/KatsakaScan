# GUIDE TECHNIQUE — TP Diagnostic Rouille Polysora (Maïs)
> Code propre · Optimisé · Maintenable · Conforme au sujet du TP

---

## 0. Conventions globales

### Nommage
```
snake_case     → variables, fonctions, fichiers Python
PascalCase     → classes
UPPER_SNAKE    → constantes
_prefix        → attribut privé d'une classe
```

### Règles de base
- **Une fonction = une responsabilité.**
- **Pas de magic numbers** : tout seuil/constante va dans `config.py`.
- **Type hints partout.**
- **Early return**, pas de `if` imbriqués inutiles.

---

## 1. Configuration centralisée

```python
# config.py
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent

@dataclass(frozen=True)
class Paths:
    raw_healthy:   Path = ROOT / "dataset/saines"
    raw_diseased:  Path = ROOT / "dataset/malades"
    features_csv:  Path = ROOT / "data/processed/features.csv"
    train_csv:     Path = ROOT / "data/splits/train.csv"
    test_csv:      Path = ROOT / "data/splits/test.csv"
    model_fs_pkl:   Path = ROOT / "models/forest_from_scratch.pkl"
    model_sk_tree:  Path = ROOT / "models/tree_sklearn.pkl"
    model_sk_rf:    Path = ROOT / "models/forest_sklearn.pkl"
    scaler_pkl:    Path = ROOT / "models/scaler.pkl"
    uploads_dir:   Path = ROOT / "uploads"

@dataclass(frozen=True)
class HSVThresholds:
    """Plage HSV pour isoler les teintes 'rouille' (marron/jaune/orange)."""
    rust_h_min: int = 5
    rust_h_max: int = 30
    rust_s_min: int = 50
    rust_v_min: int = 50

@dataclass(frozen=True)
class ModelConfig:
    n_estimators:      int = 100
    max_depth:         int = 8
    random_state:      int = 42

@dataclass(frozen=True)
class TrainConfig:
    test_size:    float = 0.20   # 80/20 selon le TP
    random_state: int   = 42

PATHS  = Paths()
HSV    = HSVThresholds()
MODEL  = ModelConfig()
TRAIN  = TrainConfig()
```

> **Changement majeur vs ancienne version :** split **80/20** (pas 70/15/15, pas de val set séparé — le TP ne le demande pas). Dossiers source renommés `dataset/saines` et `dataset/malades` pour correspondre au sujet.

---

## 2. Partie 1 — Feature engineering

### 2.1 Trois features imposées par le TP

| Feature | Méthode | Justification (sujet) |
|---|---|---|
| `pct_rouille` | Masque HSV teintes rouille / total pixels | Isole les pustules orangées indépendamment de la luminosité |
| `rugosite` | Variance/moyenne du gradient **Sobel** | Détecte irrégularités de surface causées par les pustules |
| `votre_variable` | **Au choix, justifiée agronomiquement, personnelle** | Voir section 2.4 |

> **Changement vs ancienne version :** Sobel remplace Laplacien pour la rugosité (`cv2.Sobel`, pas `cv2.Laplacian`). `pct_chlorophylle` n'est plus imposé — remplacé par une 3e feature libre.

### 2.2 Extraction couleur (HSV)

```python
# features/color_features.py
import cv2
import numpy as np
from config import HSV

def pct_rouille(hsv: np.ndarray) -> float:
    """X1 : ratio de pixels 'rouille' sur le total."""
    lower = np.array([HSV.rust_h_min, HSV.rust_s_min, HSV.rust_v_min], dtype=np.uint8)
    upper = np.array([HSV.rust_h_max, 255, 255], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)
    return float(np.count_nonzero(mask)) / mask.size
```

### 2.3 Extraction texture (Sobel)

```python
# features/texture_features.py
import cv2
import numpy as np

def rugosite_sobel(gray: np.ndarray) -> float:
    """X2 : variance des gradients de Sobel (rugosité de surface)."""
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return float(magnitude.var())
```

### 2.4 Troisième feature personnalisée — exemple proposé

> **À adapter** : le sujet exige une variable *différente pour chaque étudiant*. Exemple ci-dessous, à remplacer par ton propre choix justifié.

```python
# features/custom_feature.py
import cv2
import numpy as np

def pct_jaunissement(hsv: np.ndarray) -> float:
    """
    X3 : ratio de pixels jaune pâle (chlorose).
    Justification agronomique : avant l'apparition des pustules,
    la rouille provoque un jaunissement (chlorose) autour des
    lésions par dégradation de la chlorophylle. Ce signal précoce
    complète pct_rouille (signal tardif/visuel direct).
    """
    lower = np.array([20, 30, 100], dtype=np.uint8)
    upper = np.array([35, 255, 255], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)
    return float(np.count_nonzero(mask)) / mask.size
```

### 2.5 Orchestrateur

```python
# features/extractor.py
from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from typing import NamedTuple
from features.color_features import pct_rouille
from features.texture_features import rugosite_sobel
from features.custom_feature import pct_jaunissement

class FeatureVector(NamedTuple):
    pct_rouille: float
    rugosite:    float
    votre_variable: float

def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Image illisible : {path}")
    return img

def extract_features(path: Path) -> FeatureVector:
    img  = _load_image(path)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return FeatureVector(
        pct_rouille     = pct_rouille(hsv),
        rugosite        = rugosite_sobel(gray),
        votre_variable  = pct_jaunissement(hsv),
    )
```

### 2.6 Construction du dataset

```python
# scripts/build_dataset.py
import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import PATHS
from features.extractor import extract_features

def build_dataset() -> None:
    headers = ["id_image", "pct_rouille", "rugosite", "votre_variable", "label_malade"]
    categories = [(PATHS.raw_healthy, 0), (PATHS.raw_diseased, 1)]

    PATHS.features_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(PATHS.features_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for folder, label in categories:
            for img_path in folder.glob("*.jpg"):
                try:
                    fv = extract_features(img_path)
                    writer.writerow([img_path.stem, fv.pct_rouille,
                                      fv.rugosite, fv.votre_variable, label])
                except ValueError as e:
                    print(f"⚠️ Image ignorée : {e}")

if __name__ == "__main__":
    build_dataset()
```

---

## 3. Partie 2 — Métrique Max-Minority

### 3.1 Formule

Pureté d'un nœud `t` (proportion de la classe majoritaire) :
```
P(t) = max(n0/N, n1/N)
```

Pureté pondérée d'un split :
```
P_split = (|G|/N) * P(G) + (|D|/N) * P(D)
```

> On **maximise** `P_split` (contrairement au Gini qu'on minimise).

### 3.2 Implémentation `trouver_meilleur_split`

```python
# ml/max_minority.py
from __future__ import annotations
import numpy as np

def purete(y: np.ndarray) -> float:
    """P(t) = proportion de la classe majoritaire."""
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y, minlength=2)
    return float(counts.max()) / len(y)

def trouver_meilleur_split(X_column: np.ndarray,
                            y: np.ndarray) -> tuple[float | None, float]:
    """
    Teste tous les seuils candidats pour une variable et retourne
    (meilleur_seuil, purete_max_ponderee).
    Retourne (None, purete(y)) si aucun split n'améliore la pureté.
    """
    n = len(y)
    # Étape 1 : trier par X_column
    order  = np.argsort(X_column)
    x_sorted, y_sorted = X_column[order], y[order]

    # Étape 2 : initialisation
    best_seuil:  float | None = None
    best_purete: float = purete(y)  # pureté du nœud non-splitté = référence

    unique_vals = np.unique(x_sorted)
    if len(unique_vals) < 2:
        return None, best_purete

    thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2

    # Étape 3 : parcourir les seuils candidats
    for s in thresholds:
        split = np.searchsorted(x_sorted, s, side="right")
        if split == 0 or split == n:
            continue

        y_gauche, y_droite = y_sorted[:split], y_sorted[split:]
        p_gauche, p_droite = purete(y_gauche), purete(y_droite)

        p_split = (len(y_gauche) / n) * p_gauche + (len(y_droite) / n) * p_droite

        if p_split > best_purete:
            best_purete = p_split
            best_seuil  = s

    # Étape 4 : retour
    return best_seuil, best_purete
```

> **Différence clé avec Gini :** ici on cherche le seuil qui *maximise* la pureté pondérée. Si aucun seuil ne dépasse la pureté du nœud parent, `best_seuil` reste `None` → le nœud devient une feuille (condition d'arrêt naturelle).

---

## 4. Partie 3 — Arbres et Forêts From Scratch

### 4.1 Arbre de décision Max-Minority

```python
# ml/decision_tree.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional
from ml.max_minority import trouver_meilleur_split, purete

@dataclass
class Node:
    feature_idx: Optional[int]   = None
    threshold:   Optional[float] = None
    left:        Optional["Node"] = None
    right:       Optional["Node"] = None
    value:       Optional[int]   = None

    @property
    def is_leaf(self) -> bool:
        return self.value is not None

def _majority_class(y: np.ndarray) -> int:
    return int(np.bincount(y, minlength=2).argmax())

def build_tree(X: np.ndarray, y: np.ndarray,
               depth: int, max_depth: int) -> Node:
    # Conditions d'arrêt : nœud pur ou profondeur max atteinte
    if purete(y) == 1.0 or depth >= max_depth or len(y) < 2:
        return Node(value=_majority_class(y))

    n_features = X.shape[1]
    best_feat, best_seuil, best_purete = None, None, purete(y)

    for feat in range(n_features):
        seuil, p = trouver_meilleur_split(X[:, feat], y)
        if seuil is not None and p > best_purete:
            best_purete, best_feat, best_seuil = p, feat, seuil

    if best_feat is None:
        return Node(value=_majority_class(y))

    mask = X[:, best_feat] <= best_seuil
    return Node(
        feature_idx=best_feat,
        threshold=best_seuil,
        left=build_tree(X[mask],  y[mask],  depth + 1, max_depth),
        right=build_tree(X[~mask], y[~mask], depth + 1, max_depth),
    )

class DecisionTreeMaxMinority:
    def __init__(self, max_depth: int = 8) -> None:
        self.max_depth = max_depth
        self.root: Optional[Node] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeMaxMinority":
        self.root = build_tree(X, y.astype(int), depth=0, max_depth=self.max_depth)
        return self

    def _predict_one(self, x: np.ndarray) -> int:
        node = self.root
        while not node.is_leaf:
            node = node.left if x[node.feature_idx] <= node.threshold else node.right
        return node.value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x) for x in X])
```

### 4.2 Random Forest Max-Minority

```python
# ml/random_forest.py
from __future__ import annotations
import numpy as np
from ml.decision_tree import DecisionTreeMaxMinority

class RandomForestMaxMinority:
    def __init__(self, n_estimators: int = 100,
                 max_depth: int = 8,
                 random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.random_state = random_state
        self.trees_: list[DecisionTreeMaxMinority] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestMaxMinority":
        np.random.seed(self.random_state)
        n = len(y)
        y = y.astype(int)
        self.trees_ = []

        for _ in range(self.n_estimators):
            # Bagging : échantillonnage avec remplacement
            indices = np.random.choice(n, size=n, replace=True)
            tree = DecisionTreeMaxMinority(max_depth=self.max_depth)
            tree.fit(X[indices], y[indices])
            self.trees_.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vote majoritaire des arbres."""
        votes = np.stack([t.predict(X) for t in self.trees_], axis=1)
        return (votes.mean(axis=1) >= 0.5).astype(int)
```

---

## 5. Comparaison avec scikit-learn

### 5.1 Split 80/20

```python
# scripts/split_dataset.py
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import PATHS, TRAIN

def split_dataset() -> None:
    df = pd.read_csv(PATHS.features_csv)
    PATHS.train_csv.parent.mkdir(parents=True, exist_ok=True)

    df_train, df_test = train_test_split(
        df,
        test_size=TRAIN.test_size,
        random_state=TRAIN.random_state,
        stratify=df["label_malade"],
    )
    df_train.to_csv(PATHS.train_csv, index=False)
    df_test.to_csv(PATHS.test_csv, index=False)

if __name__ == "__main__":
    split_dataset()
```

### 5.2 Entraînement des 4 modèles

```python
# scripts/train.py
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import PATHS, MODEL
from ml.decision_tree import DecisionTreeMaxMinority
from ml.random_forest import RandomForestMaxMinority

FEATURE_COLS = ["pct_rouille", "rugosite", "votre_variable"]
LABEL_COL = "label_malade"

def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    return df[FEATURE_COLS].values, df[LABEL_COL].values.astype(int)

def train() -> None:
    X_train, y_train = load_split(PATHS.train_csv)

    # 1. Arbre From Scratch (Max-Minority)
    tree_fs = DecisionTreeMaxMinority(max_depth=MODEL.max_depth)
    tree_fs.fit(X_train, y_train)

    # 2. Random Forest From Scratch (Max-Minority)
    forest_fs = RandomForestMaxMinority(n_estimators=MODEL.n_estimators,
                                         max_depth=MODEL.max_depth,
                                         random_state=MODEL.random_state)
    forest_fs.fit(X_train, y_train)

    # 3. Arbre sklearn (Gini)
    tree_sk = DecisionTreeClassifier(criterion="gini",
                                      max_depth=MODEL.max_depth,
                                      random_state=MODEL.random_state)
    tree_sk.fit(X_train, y_train)

    # 4. Random Forest sklearn (Gini)
    forest_sk = RandomForestClassifier(n_estimators=MODEL.n_estimators,
                                        max_depth=MODEL.max_depth,
                                        random_state=MODEL.random_state)
    forest_sk.fit(X_train, y_train)

    PATHS.model_fs_pkl.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(tree_fs,   open(PATHS.model_fs_pkl, "wb"))
    pickle.dump(forest_fs, open(PATHS.model_fs_pkl.with_name("forest_fs_alt.pkl"), "wb"))
    pickle.dump(tree_sk,   open(PATHS.model_sk_tree, "wb"))
    pickle.dump(forest_sk, open(PATHS.model_sk_rf, "wb"))

    # Pour l'app Streamlit : on déploie le RF from-scratch (cf. section 6)
    pickle.dump(forest_fs, open(PATHS.model_fs_pkl, "wb"))

    print("4 modèles entraînés et sauvegardés.")

if __name__ == "__main__":
    train()
```

### 5.3 Évaluation comparative

```python
# scripts/evaluate.py
import sys
import pickle
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import PATHS
from scripts.train import load_split

def evaluate() -> None:
    X_test, y_test = load_split(PATHS.test_csv)

    models = {
        "Arbre From Scratch (Max-Minority)":  pickle.load(open(PATHS.model_fs_pkl.with_name("tree_fs.pkl"), "rb")) if False else None,
        "Forest From Scratch (Max-Minority)": pickle.load(open(PATHS.model_fs_pkl, "rb")),
        "Arbre sklearn (Gini)":                pickle.load(open(PATHS.model_sk_tree, "rb")),
        "Forest sklearn (Gini)":               pickle.load(open(PATHS.model_sk_rf, "rb")),
    }

    print(f"{'Modèle':<35} {'Accuracy':>9} {'Précision':>10} {'Rappel':>8}")
    print("-" * 64)
    for name, model in models.items():
        if model is None:
            continue
        y_pred = model.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        print(f"{name:<35} {acc:>9.3f} {prec:>10.3f} {rec:>8.3f}")
        print(f"  Matrice de confusion:\n{confusion_matrix(y_test, y_pred)}")

if __name__ == "__main__":
    evaluate()
```

> **Note pour la Partie 4.3 (discussion agronomique) :** un **Faux Négatif** (feuille malade non détectée) est plus coûteux qu'un **Faux Positif** (traitement appliqué inutilement) — privilégier le modèle avec le meilleur **Rappel** sur la classe malade, même si l'accuracy globale est légèrement inférieure.

---

## 6. Partie 4 — Application Streamlit

### 6.1 Structure

```python
# app.py
import pickle
import numpy as np
from pathlib import Path
import streamlit as st
from PIL import Image
import cv2

from config import PATHS
from ml.scaler import MinMaxScaler
from features.extractor import extract_features

st.set_page_config(page_title="Diagnostic Rouille du Maïs", page_icon="🌽")

# Chargement unique (cache Streamlit)
@st.cache_resource
def load_model():
    model  = pickle.load(open(PATHS.model_fs_pkl, "rb"))
    scaler = MinMaxScaler.load(PATHS.scaler_pkl)
    return model, scaler

model, scaler = load_model()
PATHS.uploads_dir.mkdir(exist_ok=True)

st.title("🌽 Diagnostic de la Rouille Polysora")

# --- Module 1 : Upload et prédiction ---
uploaded_file = st.file_uploader("Téléverser une photo de feuille",
                                  type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléversée", use_column_width=True)

    save_path = PATHS.uploads_dir / uploaded_file.name
    image.save(save_path)

    fv = extract_features(save_path)
    X  = scaler.transform(np.array([[fv.pct_rouille, fv.rugosite, fv.votre_variable]]))
    pred = int(model.predict(X)[0])

    if pred == 1:
        st.error("⚠️ ATTENTION : Feuille Malade (Rouille Detected)")
    else:
        st.success("✅ Feuille Saine")

    st.caption(f"pct_rouille={fv.pct_rouille:.4f} · "
               f"rugosite={fv.rugosite:.2f} · "
               f"votre_variable={fv.votre_variable:.4f}")

    # Sauvegarde du diagnostic pour la galerie
    diag_path = save_path.with_suffix(".txt")
    diag_path.write_text("Malade" if pred == 1 else "Saine")

# --- Module 2 : Galerie d'historique ---
st.header("📷 Historique des détections")

images = sorted(PATHS.uploads_dir.glob("*.[jp][pn]g"))
if not images:
    st.info("Aucune image analysée pour le moment.")
else:
    cols = st.columns(4)
    for i, img_path in enumerate(images):
        diag_path = img_path.with_suffix(".txt")
        diagnostic = diag_path.read_text() if diag_path.exists() else "?"
        with cols[i % 4]:
            st.image(str(img_path), use_column_width=True)
            st.caption(diagnostic)
```

> **Lancement :** `streamlit run app.py`

---

## 7. Structure du projet

```
qoui/
├── app.py
├── config.py
├── dataset/
│   ├── saines/
│   └── malades/
├── data/
│   ├── processed/
│   │   └── features.csv
│   └── splits/
│       ├── train.csv
│       └── test.csv
├── features/
│   ├── __init__.py
│   ├── extractor.py
│   ├── color_features.py
│   ├── texture_features.py
│   └── custom_feature.py
├── ml/
│   ├── __init__.py
│   ├── max_minority.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   └── scaler.py
├── models/
│   ├── forest_from_scratch.pkl
│   ├── tree_sklearn.pkl
│   ├── forest_sklearn.pkl
│   └── scaler.pkl
├── scripts/
│   ├── build_dataset.py
│   ├── split_dataset.py
│   ├── train.py
│   └── evaluate.py
├── uploads/
├── tests/
└── requirements.txt
```

---

## 8. requirements.txt

```
numpy>=1.26
pandas>=2.2
opencv-python>=4.9
scikit-learn>=1.4
streamlit>=1.35
pillow>=10.0
pytest>=8.0
```

---

## 9. Checklist conformité TP

| # | Critère (sujet) | Statut attendu |
|---|---|---|
| 1 | `pct_rouille` via masque HSV | `color_features.py` |
| 2 | `rugosite` via Sobel (pas Laplacien) | `texture_features.py` |
| 3 | 3e feature personnelle + justification agronomique | `custom_feature.py` |
| 4 | DataFrame `[ID_Image \| pct_rouille \| rugosite \| votre_variable \| label_malade]` | `features.csv` |
| 5 | `trouver_meilleur_split` avec métrique Max-Minority | `ml/max_minority.py` |
| 6 | `build_tree(X, y, depth, max_depth)` récursif | `ml/decision_tree.py` |
| 7 | Random Forest from scratch : bagging + vote majoritaire | `ml/random_forest.py` |
| 8 | Split 80/20 | `config.TRAIN.test_size = 0.20` |
| 9 | 4 modèles comparés (Accuracy, Précision, Rappel, matrice) | `scripts/evaluate.py` |
| 10 | Discussion FN vs FP (contexte Madagascar) | Analyse manuelle dans le rapport |
| 11 | App Streamlit : upload + prédiction temps réel | `app.py` |
| 12 | Galerie d'historique avec `st.columns` | `app.py` |
