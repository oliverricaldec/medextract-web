"""
Entrena APF1 (CRF + rasgos manuales) en tu PC, sin GPU (~1-2 min).

    python entrenar_apf1_local.py
"""

import joblib
import sklearn_crfsuite

from datos import metricas_estrictas, preparar_particiones
from medextract import CARPETA_MODELOS, rasgos_oracion

train, val, test = preparar_particiones()

crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100,
                           all_possible_transitions=True)
crf.fit([rasgos_oracion(o) for o in train["tokens"]], train["etiquetas"].tolist())

pred = crf.predict([rasgos_oracion(o) for o in test["tokens"]])
print("TEST APF1 (F1 estricto)")
print(metricas_estrictas(test["etiquetas"].tolist(), pred).round(4).to_string(index=False))

CARPETA_MODELOS.mkdir(exist_ok=True)
destino = CARPETA_MODELOS / "crf_rasgos_manuales.joblib"
joblib.dump(crf, destino)
print("Modelo guardado en", destino)
