# Structure du projet Katsaka

```
Katsaka/
│
├── app.py                        # Application Streamlit (UI upload + prédiction)
│
├── app/                          # Dossier réservé (vide — pas de Flask/FastAPI)
│   ├── templates/
│   └── static/
│
├── dataset/                      # Images brutes + CSV générés
│   ├── saines/                   # Images de feuilles saines
│   ├── malades/                  # Images de feuilles malades
│   ├── features.csv              # Dataset extrait (features + labels)
│   ├── train.csv                 # Split entraînement (80%)
│   └── test.csv                  # Split test (20%) — pas de val set
│
├── features/
│   ├── __init__.py
│   ├── extractor.py              # Pipeline extraction → FeatureVector(3 features)
│   ├── color_features.py         # pct_rouille via masque HSV orange/marron
│   ├── texture_features.py       # rugosite via variance gradient Sobel
│   └── custom_feature.py         # pct_jaunissement via masque HSV jaune pâle
│
├── ml/
│   ├── __init__.py
│   ├── max_minority.py           # Métrique pureté Max-Minority + trouver_meilleur_split
│   ├── decision_tree.py          # DecisionTreeMaxMinority (fit, predict, predict_proba)
│   ├── random_forest.py          # RandomForestMaxMinority — bagging, pas de feature bagging
│   ├── metrics.py                # Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC
│   └── scaler.py                 # MinMaxScaler from scratch (fit, transform, save/load)
│
├── models/                       # Modèles sérialisés (.pkl)
│
├── scripts/
│   ├── build_dataset.py          # Parcourt dataset/saines|malades → features.csv
│   ├── split_dataset.py          # Génère train.csv / test.csv (80/20, pas de val)
│   ├── train.py                  # Entraîne arbre + forest (from scratch + sklearn)
│   └── evaluate.py               # Évalue sur test.csv — métriques from scratch (ml/metrics)
│
├── config.py                     # Chemins (PATHS), hyperparamètres (HP), plages HSV (HSV)
├── requirements.txt
└── README.md
```

---

## Modules clés

### `features/extractor.py`
`extract_features(image_path) -> FeatureVector` : charge image, convertit BGR→HSV+Gray, appelle les 3 sous-modules, retourne un dataclass avec `pct_rouille`, `rugosite`, `pct_jaunissement`.

### `ml/max_minority.py`
`purete(y)` = proportion classe majoritaire. `trouver_meilleur_split(col, y)` cherche le seuil (milieu entre valeurs uniques triées) qui maximise la pureté pondérée des deux nœuds fils.

### `ml/decision_tree.py`
`DecisionTreeMaxMinority` : `fit`, `predict`, **`predict_proba`** (ratio classe 1 dans la feuille atteinte). Feuilles stockent `proba` pour le calcul ROC-AUC.

### `ml/random_forest.py`
`RandomForestMaxMinority` : bagging sur lignes (bootstrap), vote majoritaire, **`predict_proba`** (moyenne des probas feuilles de chaque arbre).
⚠ Pas de feature bagging — tous les arbres voient toutes les features.

### `scripts/evaluate.py`
Charge les 4 modèles (arbre+forest × from-scratch+sklearn), calcule Accuracy / Precision / Recall / F1 / ROC-AUC avec `ml/metrics` uniquement (zéro dépendance à `sklearn.metrics`).
