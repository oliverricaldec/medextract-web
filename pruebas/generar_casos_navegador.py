"""
Genera casos de prueba para comparar el motor del navegador (motor.js) con Python:
ids del tokenizador, etiquetas de APF1 (joblib) y de APF3 (el mismo ONNX int8 que usa la web).
Salida: web_estatica/casos_prueba.json (no se publica).
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import medextract  # noqa: E402
from datos import preparar_particiones  # noqa: E402
from medextract import MAPA_ETIQUETAS, NOMBRE_BETO  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

RAIZ = Path(__file__).parent.parent
tok = AutoTokenizer.from_pretrained(NOMBRE_BETO)
apf1 = medextract.APF1(str(RAIZ / "modelos" / "crf_rasgos_manuales.joblib"))
crf3 = json.load(open(RAIZ / "web_estatica" / "modelo" / "apf3_crf.json"))
sesion = ort.InferenceSession(str(RAIZ / "web_estatica" / "modelo" / "apf3.onnx"), providers=["CPUExecutionProvider"])

_, _, test = preparar_particiones()
textos = [" ".join(map(str, t)) for t in test["tokens"][:200]]
textos += ["El paciente presenta diabetes mellitus tipo 2 y se le prescribe metformina de 850mg.",
           "¿Hipertensión? ÚLCERA gástrica, (dolor) 3x/día; COVID-19 + PCR... Ñandú ü ç € ½ 中文"]


def viterbi(em):
    ini, fin, trans = (torch.tensor(crf3[k]) for k in ("inicio", "fin", "trans"))
    dp, hist = ini + em[0], []
    for t in range(1, len(em)):
        mejor, origen = (dp.unsqueeze(1) + trans).max(0)
        dp = mejor + em[t]
        hist.append(origen)
    ult = int((dp + fin).argmax())
    ruta = [ult]
    for o in reversed(hist):
        ult = int(o[ult]); ruta.append(ult)
    return ruta[::-1]


def apf3(oracion):
    ids, primeras = [tok.cls_token_id], []
    for p in oracion:
        piezas = tok.encode(p, add_special_tokens=False) or [tok.unk_token_id]
        if len(ids) + len(piezas) > 127:
            break
        primeras.append(len(ids)); ids += piezas
    ids.append(tok.sep_token_id)
    em = sesion.run(None, {"input_ids": np.array([ids]), "attention_mask": np.ones((1, len(ids)), dtype=np.int64)})[0][0]
    etq = [MAPA_ETIQUETAS[i] for i in viterbi(torch.tensor(em[primeras]))]
    return etq + ["O"] * (len(oracion) - len(etq))


casos = []
for texto in textos:
    oraciones = medextract.partir_oraciones(medextract.tokenizar(texto))
    palabras = [p[0] for o in oraciones for p in o]
    casos.append({
        "texto": texto,
        "ids": {p: (tok.encode(p, add_special_tokens=False) or [tok.unk_token_id]) for p in palabras},
        "apf1": [e for o in oraciones for e in apf1.predecir([[p[0] for p in o]])[0]],
        "apf3": [e for o in oraciones for e in apf3([p[0] for p in o])],
    })

json.dump(casos, open(RAIZ / "web_estatica" / "casos_prueba.json", "w", encoding="utf-8"), ensure_ascii=False)
print("Casos:", len(casos))
