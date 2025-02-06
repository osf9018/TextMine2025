# TextMine2025

Code pour reproduire le système gagnant du défi TextMine 2025 : https://www.kaggle.com/competitions/defi-text-mine-2025

> Adaptation d'un modèle de langue encodeur-décodeur pour l'extraction de relations dans des rapports de renseignement. Adrien Guille. Atelier Fouille de Textes (TextMine @ EGC), Strasbourg (France), 2025

## Génération du jeu de données

- Télécharger les fichiers `train.csv` et `test_01-07-2024.csv`
- Générer le jeu de données à l'aide du notebook `Dataset.ipynb`

## Ajustement (LoRA) d'un encodeur-décodeur pré-entraîné

- Exécuter l'instruction `python finetune.py --data dataset --checkpoint mt0-xxl --modules qv --rank 64 --batch_size 16 --n_epochs 2` pour spécialiser le modèle `mt0-xxl` pré-entraîné par [BigScience](https://huggingface.co/bigscience) (ou d'autres variantes de mt0, ou bien des variantes de flan-t5 pré-entraînées par [Google](https://huggingface.co/collections/google/flan-t5-release-65005c39e3201fff885e22fb)). Seules les matrices $Q$ et $V$ sont ajustées, pour un rang de 64.
  - Ajuster les arguments `batch_size` et `gradient_accumulation` de sorte que leur produit soit égal à 16 pour reproduire les résultats obtenus à la compétition

## Utilisation de l'encodeur-décodeur

Le modèle ajusté est disponible sur HuggingFace : https://huggingface.co/AdrienGuille/TextMine2025. Le code ci-dessous montre comment utiliser le modèle pour prédire l'existence d'une relation dans un texte :

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model = AutoModelForSeq2SeqLM.from_pretrained(
        "AdrienGuille/TextMine2025",
        torch_dtype=torch.bfloat16, # requires a compatible GPU, otherwise should be set to torch.float16
        return_dict=True,
        device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("bigscience/mt0-xxl")

# format the prompt according to the following template
prompt = """Does the relation (head_entity: [Constance Dupuis], relation_type: is_in_contact_with},
tail_entity: [Airîle, compagnie aérienne]), exists in the following text: "L’avion NY8 de la
compagnie Airîle a lancé sa dernière position via le signal radio avant de se crasher dans une forêt
en Malaisie le 19 février 2003. La compagnie aérienne a alerté les secours pour évacuer les
passagers. Les hélicoptères d’urgence ont retrouvé l’appareil en feu. Les autorités malaisiennes ont
recensé 15 morts au total. Cet incident n’a fait que peu de survivants, dont Constance Dupuis,
présidente de l’association « des médicaments pour tous » en Grèce. D’après son témoignage, le
NY8 a connu une défaillance technique que les pilotes n’ont pas pu contrôler. Les corps ont été
transportés par brancard à la morgue."?"""

# do not do sample for generation
input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
output_ids = model.generate(**input_ids, num_beams=1, do_sample=False, max_new_tokens=4)

# output should be either yer or no
answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
prediction = "yes" in answer.lower()
```
