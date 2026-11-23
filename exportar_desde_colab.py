# ============================================================
# CELDA PARA PEGAR AL FINAL DEL NOTEBOOK DE COLAB
# Exporta los modelos entrenados (APF1, APF2, APF3) en el formato
# que espera la web y descarga un .zip para copiarlo a ./modelos
# ============================================================
import shutil
from pathlib import Path

SALIDA = Path("/content/modelos")
SALIDA.mkdir(exist_ok=True)

# APF1 — CRF con rasgos manuales
joblib.dump(crf_manual, SALIDA / "crf_rasgos_manuales.joblib")
print("APF1 exportado")

# APF2 — BETO fine-tuned (modelo + tokenizador)
if "trainer_beto" in globals():
    trainer_beto.save_model(str(SALIDA / "apf2_beto_finetuned"))
    tokenizador_rapido.save_pretrained(str(SALIDA / "apf2_beto_finetuned"))
    print("APF2 exportado")

# APF3 — BETO + CRF neuronal (mejor checkpoint por validación)
if "apf3_mejor" in globals():
    shutil.copy(apf3_mejor["checkpoint"], SALIDA / "apf3_beto_crf.pt")
    print("APF3 exportado (semilla", apf3_mejor["semilla"], ")")

shutil.make_archive("/content/modelos_medextract", "zip", SALIDA)

# Opción A: descargar directamente
from google.colab import files
files.download("/content/modelos_medextract.zip")

# Opción B (si la descarga falla por tamaño, ~1 GB con APF2 + APF3): copiar a Drive
# from google.colab import drive; drive.mount("/content/drive")
# shutil.copy("/content/modelos_medextract.zip", "/content/drive/MyDrive/MedExtract/")
