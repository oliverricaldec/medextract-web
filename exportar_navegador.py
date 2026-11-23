"""
Exporta los modelos para que corran 100 % en el navegador (Space estático de Hugging Face):

  web_estatica/modelo/apf3.onnx      BETO + capa lineal (emisiones), cuantizado int8
  web_estatica/modelo/apf3_crf.json  inicio / fin / transiciones del CRF (con reglas BIOES)
  web_estatica/modelo/vocab.txt      vocabulario WordPiece de BETO
  web_estatica/modelo/apf1.json      pesos del CRF de rasgos manuales

Después mide el F1 del modelo exportado en el test, con la misma tokenización que usa el navegador.

    python exportar_navegador.py
"""

import json
import sys

import joblib
import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoTokenizer

import medextract
from medextract import CARPETA_MODELOS, MAPA_ETIQUETAS, NOMBRE_BETO

SALIDA = medextract.Path(__file__).parent / "web_estatica" / "modelo"
SALIDA.mkdir(parents=True, exist_ok=True)

# ---------------- APF3 -> ONNX ----------------
apf3 = medextract.APF3(str(CARPETA_MODELOS / "apf3_beto_crf.pt"))


class Emisiones(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.beto, self.clasificador = m.beto, m.clasificador

    def forward(self, input_ids, attention_mask):
        return self.clasificador(self.beto(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)


envoltura = Emisiones(apf3.modelo).eval()
ejemplo = apf3.tokenizador(["El", "paciente", "tiene", "fiebre"], is_split_into_words=True, return_tensors="pt")
ruta_fp32 = SALIDA / "apf3_fp32.onnx"
torch.onnx.export(
    envoltura, (ejemplo["input_ids"], ejemplo["attention_mask"]), str(ruta_fp32),
    input_names=["input_ids", "attention_mask"], output_names=["emisiones"],
    dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "emisiones": {0: "b", 1: "s"}},
    opset_version=17, dynamo=False,
)
ruta_int8 = SALIDA / "apf3.onnx"
quantize_dynamic(str(ruta_fp32), str(ruta_int8), weight_type=QuantType.QInt8, per_channel=True)
print(f"ONNX fp32: {ruta_fp32.stat().st_size/1e6:.0f} MB | int8: {ruta_int8.stat().st_size/1e6:.0f} MB")

ini, fin, trans = (t.detach().numpy().round(5).tolist() for t in apf3.modelo.crf.puntajes())
json.dump({"etiquetas": [MAPA_ETIQUETAS[i] for i in range(13)], "inicio": ini, "fin": fin, "trans": trans},
          open(SALIDA / "apf3_crf.json", "w"))

tok = AutoTokenizer.from_pretrained(NOMBRE_BETO)
# BertNormalizer sin minúsculas ni quitar tildes + WordPiece "##" (igual que implementa app.js)
vocab = sorted(tok.get_vocab().items(), key=lambda kv: kv[1])
assert [i for _, i in vocab] == list(range(len(vocab)))
(SALIDA / "vocab.txt").write_text("\n".join(p for p, _ in vocab), encoding="utf-8", newline="\n")

# ---------------- APF1 -> JSON ----------------
crf1 = joblib.load(CARPETA_MODELOS / "crf_rasgos_manuales.joblib")
estado = {}
for (atributo, etiqueta), peso in crf1.state_features_.items():
    if abs(peso) > 1e-6:
        estado.setdefault(atributo, {})[etiqueta] = round(peso, 4)
transiciones = {}
for (origen, destino), peso in crf1.transition_features_.items():
    transiciones.setdefault(origen, {})[destino] = round(peso, 4)
json.dump({"etiquetas": list(crf1.classes_), "estado": estado, "trans": transiciones},
          open(SALIDA / "apf1.json", "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print(f"APF1 JSON: {(SALIDA / 'apf1.json').stat().st_size/1e6:.1f} MB, {len(estado)} atributos")

# ---------------- Evaluación del ONNX en test ----------------
if "--sin-evaluar" in sys.argv:
    sys.exit()

from datos import metricas_estrictas, preparar_particiones  # noqa: E402

_, _, test = preparar_particiones()
crf = apf3.modelo.crf


def ids_por_palabra(palabras):
    """Igual que el navegador: cada palabra se tokeniza por separado y se unen con [CLS]/[SEP]."""
    ids, primeras = [tok.cls_token_id], []
    for p in palabras:
        piezas = tok.encode(p, add_special_tokens=False) or [tok.unk_token_id]
        if len(ids) + len(piezas) > 127:
            break
        primeras.append(len(ids))
        ids.extend(piezas)
    return ids + [tok.sep_token_id], primeras


def evaluar(ruta):
    sesion = ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
    pred = []
    for palabras in test["tokens"]:
        palabras = list(palabras)
        ids, primeras = ids_por_palabra(palabras)
        em = sesion.run(None, {"input_ids": np.array([ids], dtype=np.int64),
                               "attention_mask": np.ones((1, len(ids)), dtype=np.int64)})[0][0]
        ruta_tags = crf.decode(torch.tensor(em[primeras]))
        etiquetas = [MAPA_ETIQUETAS[i] for i in ruta_tags]
        pred.append(etiquetas + ["O"] * (len(palabras) - len(etiquetas)))
    rep = metricas_estrictas(test["etiquetas"].tolist(), pred)
    return rep


for nombre, ruta in (("fp32", ruta_fp32), ("int8", ruta_int8)):
    rep = evaluar(ruta)
    print(f"\nTEST ONNX {nombre}\n", rep.round(4).to_string(index=False))

ruta_fp32.unlink()
