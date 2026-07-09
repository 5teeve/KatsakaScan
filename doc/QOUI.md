# QOUI — Détection de maladie sur feuille de maïs

## Contexte
Application web Python qui analyse une image de feuille de maïs et prédit si la plante est **saine ou malade** via un Random Forest implémenté from scratch.

## Input / Output
- **Input :** image JPG/PNG d'une feuille de maïs
- **Output :** `0` = Saine, `1` = Malade + score de confiance

## Pipeline

### Données
Chaque image est convertie en vecteur de 3 features :

| Feature | Méthode |
|---|---|
| `pct_rouille` | % pixels dans plage HSV orange/marron (pustules de rouille) |
| `rugosite` | Variance du gradient **Sobel** (texture foliaire) |
| `pct_jaunissement` | % pixels jaune pâle HSV (chlorose précoce avant pustules) |

Format : `id_image | pct_rouille | rugosite | pct_jaunissement | label_malade`
Split : **80% train / 20% test** (pas de val set)

### Algorithme
- **Pureté :** métrique **Max-Minority** (proportion classe majoritaire dans le nœud)
- **Arbre de décision :** split récursif sur le seuil maximisant la pureté pondérée Max-Minority
- **Random Forest :** bagging de N arbres (bootstrap sur lignes), vote majoritaire / moyenne des probabilités feuille
- Note : pas de feature bagging (toutes les features évaluées à chaque split)

### Métriques (from scratch)
`ml/metrics.py` implémente : Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC (méthode trapézoïdale).
Comparaison modèles from scratch vs `sklearn` dans `scripts/evaluate.py`.

## Stack
Python · OpenCV · NumPy · **Streamlit** · scikit-learn (comparaison uniquement)
