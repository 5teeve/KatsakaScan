# QOUI
## Contexte
- Une application pour detecter si une plante de maïs est malade ou non
## Resultat a attendre
- Dit si la plante est malade ou non
## Input
- Image feuille maïs

# COMMENT
## Stack
- Python (web)
## Pipeline
### Collecte de donnees
- Convertir en tableau (colonne, ligne)
- Chercher les caracteristiques a prendre:
    - HSV
    - Rugosite
    - (une autre caracteristique verifier agronomiquement)
- Format donnée:
    id_image | pct_rouille | rugosite | l'autre caracteristique | label_malade (0 ou 1)
### Algorithme
- Notion pureté
- Random forest: from scratch
- Arbre de décision
### Entrainement
### Test
- Comparer avec celle de skitlearn