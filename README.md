# TextMine2025

Code pour reproduire la système gagnant du défi TextMine 2025 : https://www.kaggle.com/competitions/defi-text-mine-2025

> Adaptation d'un modèle de langue encodeur-décodeur pour l'extraction de relations dans des rapports de renseignement. Adrien Guille. Atelier Fouille de Textes (TextMine @ EGC), Strasbourg (France), 2025

## Jeu de données

- Télécharger les fichiers `train.csv` et `test_01-07-2024.csv`
- Générer le jeu de données à l'aide du notebook `Dataset.ipynb`

## Spécialisation (LoRA) d'un encodeur-décodeur pré-entraîné

- Exécuter l'instruction `python finetune.py --data dataset --checkpoint mt0-xxl --modules qv --rank 64 --batch_size 16 --n_epochs 2` pour spécialiser le modèle `mt0-xxl` pré-entraîné par [BigScience](https://huggingface.co/bigscience) (ou d'autres variantes de mt0, ou bien des variantes de flan-t5 pré-entraînées par [Google](https://huggingface.co/collections/google/flan-t5-release-65005c39e3201fff885e22fb)). Seules les matrices $Q$ et $V$ sont ajustées, pour un rang de 64.
  - Ajuster les arguments `batch_size` et `gradient_accumulation` de sorte que leur produit soit égal à 16 pour reproduire les résultats obtenus à la compétition
