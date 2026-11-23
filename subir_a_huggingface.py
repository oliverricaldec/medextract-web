"""
Publica MedExtract en Hugging Face (gratis):

  - Repo de modelo  <usuario>/medextract-modelos : apf3.onnx, apf3_crf.json, vocab.txt, apf1.json
  - Space estático  <usuario>/medextract         : index.html + motor.js (los modelos corren en el navegador)

    hf auth login --token TU_TOKEN     # una sola vez
    python exportar_navegador.py       # genera web_estatica/modelo/
    python subir_a_huggingface.py
"""

from pathlib import Path

from huggingface_hub import HfApi

RAIZ = Path(__file__).parent / "web_estatica"

README_SPACE = """---
title: MedExtract
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: static
pinned: false
short_description: Extracción de entidades médicas en español (BETO + CRF)
---

# MedExtract

Extracción de entidades clínicas en español (**PROBLEM**, **TREATMENT**, **TEST**).
Los modelos (APF1: CRF + rasgos manuales; APF3: BETO + CRF neuronal) se ejecutan en el navegador
con ONNX Runtime Web. Proyecto universitario: no usar para decisiones médicas reales.
"""

README_MODELOS = """---
language: es
tags: [token-classification, ner, medical, onnx]
---

# MedExtract — modelos para el navegador

- `apf3.onnx`: BETO (dccuchile/bert-base-spanish-wwm-cased) fine-tuned + capa lineal, cuantizado int8. Devuelve emisiones por sub-token.
- `apf3_crf.json`: puntajes de inicio, fin y transición del CRF con reglas BIOES (decodificar con Viterbi).
- `vocab.txt`: vocabulario WordPiece de BETO.
- `apf1.json`: pesos del CRF de rasgos manuales.

F1 estricto micro en test (HUMADEX/spanish_ner_dataset): APF3 0.702 (int8: 0.692), APF1 0.540.
"""

api = HfApi()
usuario = api.whoami()["name"]
repo_modelos = f"{usuario}/medextract-modelos"
repo_space = f"{usuario}/medextract"

api.create_repo(repo_modelos, repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj=README_MODELOS.encode(), path_in_repo="README.md", repo_id=repo_modelos)
api.upload_folder(folder_path=RAIZ / "modelo", repo_id=repo_modelos, commit_message="Modelos MedExtract",
                  allow_patterns=["apf3.onnx", "apf3_crf.json", "vocab.txt", "apf1.json"])

api.create_repo(repo_space, repo_type="space", space_sdk="static", exist_ok=True)
api.upload_file(path_or_fileobj=README_SPACE.encode(), path_in_repo="README.md", repo_id=repo_space, repo_type="space")
api.upload_folder(folder_path=RAIZ, repo_id=repo_space, repo_type="space", commit_message="Web MedExtract",
                  allow_patterns=["index.html", "motor.js"])

print("Space:", f"https://huggingface.co/spaces/{repo_space}")
print("Link directo:", f"https://{usuario.lower()}-medextract.static.hf.space")
