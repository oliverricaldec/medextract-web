"""
Entrena APF3 (BETO + CRF neuronal con transiciones BIOES restringidas) en tu PC con GPU.
Reproduce las celdas 56-60 del notebook: mismas particiones, hiperparámetros, 3 semillas,
early stopping por F1 de validación y evaluación final en test.

    python entrenar_apf3_local.py            # 3 semillas (como el notebook)
    python entrenar_apf3_local.py --rapido   # 1 semilla
"""

import json
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from datos import metricas_estrictas, preparar_particiones
from medextract import CARPETA_MODELOS, MAPA_ETIQUETAS, NOMBRE_BETO

CONFIG = {
    "longitud_subtokens": 128, "batch_size": 16, "epochs": 10, "paciencia": 3,
    "lr_beto": 2e-5, "lr_cabeza_crf": 1e-3, "dropout": 0.1, "weight_decay": 0.01,
    "warmup_ratio": 0.10, "semillas": [42] if "--rapido" in sys.argv else [42, 52, 62],
}
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZADOR = AutoTokenizer.from_pretrained(NOMBRE_BETO, use_fast=True)
print("Dispositivo:", DISPOSITIVO, torch.cuda.get_device_name(0) if DISPOSITIVO.type == "cuda" else "")


# ---------------- Datos (alineación a la primera subpieza) ----------------

def construir_dataset(df):
    filas = []
    for palabras, tags in zip(df["tokens"], df["tags_numericos"]):
        palabras = list(palabras)
        entrada = TOKENIZADOR(palabras, is_split_into_words=True, truncation=True,
                              max_length=CONFIG["longitud_subtokens"])
        etiquetas, prev = [], None
        for w in entrada.word_ids():
            etiquetas.append(-100 if w is None or w == prev else tags[w])
            prev = w
        filas.append({"input_ids": entrada["input_ids"], "attention_mask": entrada["attention_mask"],
                      "labels": etiquetas, "num_palabras": len(palabras)})
    return filas


def colador(items):
    datos = TOKENIZADOR.pad([{k: x[k] for k in ("input_ids", "attention_mask")} for x in items],
                            padding=True, return_tensors="pt")
    largo = datos["input_ids"].shape[1]
    datos["labels"] = torch.tensor([x["labels"] + [-100] * (largo - len(x["labels"])) for x in items])
    datos["num_palabras"] = [x["num_palabras"] for x in items]
    return datos


# ---------------- Modelo (mismas clases del notebook) ----------------

def reglas_bioes(id2label):
    n = len(id2label)
    inicio, fin = torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)
    pasos = torch.zeros((n, n), dtype=torch.bool)
    split = lambda tag: ("O", None) if tag == "O" else tag.split("-", 1)
    for i in range(n):
        p, t = split(id2label[i])
        inicio[i] = p in ("O", "B", "S")
        fin[i] = p in ("O", "E", "S")
        for j in range(n):
            q, u = split(id2label[j])
            if p in ("O", "E", "S"):
                pasos[i, j] = q in ("O", "B", "S")
            elif p in ("B", "I"):
                pasos[i, j] = q in ("I", "E") and t == u
    return inicio, fin, pasos


class APF3CRF(nn.Module):
    def __init__(self, n_tags, id2label):
        super().__init__()
        self.inicio = nn.Parameter(torch.zeros(n_tags))
        self.fin = nn.Parameter(torch.zeros(n_tags))
        self.trans = nn.Parameter(torch.empty(n_tags, n_tags))
        nn.init.uniform_(self.trans, -0.1, 0.1)
        inicio, fin, pasos = reglas_bioes(id2label)
        self.register_buffer("inicio_valido", inicio)
        self.register_buffer("fin_valido", fin)
        self.register_buffer("trans_valida", pasos)

    def puntajes(self):
        return (self.inicio.masked_fill(~self.inicio_valido, -1e4),
                self.fin.masked_fill(~self.fin_valido, -1e4),
                self.trans.masked_fill(~self.trans_valida, -1e4))

    def forward(self, emisiones, etiquetas, mascara):
        ini, fin, trans = self.puntajes()
        b, t, n = emisiones.shape
        idx = torch.arange(b, device=emisiones.device)
        score = ini[etiquetas[:, 0]] + emisiones[idx, 0, etiquetas[:, 0]]
        alpha = ini.unsqueeze(0) + emisiones[:, 0, :]
        for pos in range(1, t):
            activo = mascara[:, pos]
            score = score + (trans[etiquetas[:, pos - 1], etiquetas[:, pos]] +
                             emisiones[idx, pos, etiquetas[:, pos]]) * activo
            nuevo = torch.logsumexp(alpha.unsqueeze(2) + trans.unsqueeze(0) + emisiones[:, pos, :].unsqueeze(1), dim=1)
            alpha = torch.where(activo.unsqueeze(1), nuevo, alpha)
        ultimos = mascara.long().sum(1) - 1
        score = score + fin[etiquetas[idx, ultimos]]
        log_z = torch.logsumexp(alpha + fin.unsqueeze(0), dim=1)
        return (log_z - score).mean()

    @torch.no_grad()
    def decode(self, emisiones, mascara):
        ini, fin, trans = self.puntajes()
        resultado = []
        for em, mask in zip(emisiones, mascara):
            t = int(mask.sum().item())
            dp, historial = ini + em[0], []
            for pos in range(1, t):
                mejor, origen = (dp.unsqueeze(1) + trans).max(dim=0)
                dp = mejor + em[pos]
                historial.append(origen)
            ultimo = int((dp + fin).argmax().item())
            ruta = [ultimo]
            for origen in reversed(historial):
                ultimo = int(origen[ultimo].item())
                ruta.append(ultimo)
            resultado.append(list(reversed(ruta)))
        return resultado


class ModeloBETOCRF(nn.Module):
    def __init__(self):
        super().__init__()
        self.beto = AutoModel.from_pretrained(NOMBRE_BETO)
        self.dropout = nn.Dropout(CONFIG["dropout"])
        self.clasificador = nn.Linear(self.beto.config.hidden_size, len(MAPA_ETIQUETAS))
        self.crf = APF3CRF(len(MAPA_ETIQUETAS), MAPA_ETIQUETAS)

    def forward(self, input_ids, attention_mask, labels):
        contextual = self.beto(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        validos = labels.ne(-100)
        longitudes = validos.sum(1)
        max_p = int(longitudes.max().item())
        b, _, h = contextual.shape
        palabras = contextual.new_zeros(b, max_p, h)
        tags = labels.new_zeros(b, max_p)
        mascara = torch.arange(max_p, device=labels.device).unsqueeze(0) < longitudes.unsqueeze(1)
        for i in range(b):
            k = int(longitudes[i].item())
            palabras[i, :k] = contextual[i, validos[i]]
            tags[i, :k] = labels[i, validos[i]]
        emisiones = self.clasificador(self.dropout(palabras))
        return self.crf(emisiones, tags, mascara), emisiones, mascara


# ---------------- Entrenamiento ----------------

def fijar_semilla(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def predecir(modelo, dataset):
    modelo.eval()
    salidas = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False, collate_fn=colador):
            num = batch.pop("num_palabras")
            batch = {k: v.to(DISPOSITIVO) for k, v in batch.items()}
            _, em, mask = modelo(**batch)
            for ids, n in zip(modelo.crf.decode(em, mask), num):
                salidas.append([MAPA_ETIQUETAS[x] for x in ids] + ["O"] * (n - len(ids)))
    return salidas


def f1_micro(y_real, y_pred):
    r = metricas_estrictas(y_real, y_pred)
    return float(r.loc[r["entidad"] == "MICRO", "f1"].iloc[0])


def entrenar(semilla, ds_train, ds_val, y_val):
    fijar_semilla(semilla)
    modelo = ModeloBETOCRF().to(DISPOSITIVO)
    cargador = DataLoader(ds_train, batch_size=CONFIG["batch_size"], shuffle=True, collate_fn=colador)
    opt = AdamW([
        {"params": list(modelo.beto.parameters()), "lr": CONFIG["lr_beto"]},
        {"params": list(modelo.clasificador.parameters()) + list(modelo.crf.parameters()), "lr": CONFIG["lr_cabeza_crf"]},
    ], weight_decay=CONFIG["weight_decay"])
    total = CONFIG["epochs"] * len(cargador)
    plan = get_linear_schedule_with_warmup(opt, int(total * CONFIG["warmup_ratio"]), total)
    ruta = CARPETA_MODELOS / f"apf3_semilla_{semilla}.pt"
    mejor_f1, mejor_epoca, sin_mejora, historial = -1.0, 0, 0, []

    for epoca in range(1, CONFIG["epochs"] + 1):
        modelo.train(); perdidas = []; t0 = time.time()
        for batch in cargador:
            batch.pop("num_palabras")
            batch = {k: v.to(DISPOSITIVO) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            perdida, _, _ = modelo(**batch)
            perdida.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step(); plan.step()
            perdidas.append(float(perdida.item()))
        f1 = f1_micro(y_val, predecir(modelo, ds_val))
        historial.append({"semilla": semilla, "epoca": epoca, "loss_train": float(np.mean(perdidas)), "f1_val": f1})
        print(f"Semilla={semilla} época={epoca} pérdida={np.mean(perdidas):.4f} F1 val={f1:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if f1 > mejor_f1 + 1e-6:
            mejor_f1, mejor_epoca, sin_mejora = f1, epoca, 0
            torch.save(modelo.state_dict(), ruta)
        else:
            sin_mejora += 1
            if sin_mejora >= CONFIG["paciencia"]:
                print("Parada temprana: 3 épocas sin mejora", flush=True)
                break
    del modelo
    torch.cuda.empty_cache()
    return {"semilla": semilla, "mejor_f1_val": mejor_f1, "mejor_epoca": mejor_epoca,
            "checkpoint": str(ruta), "historial": historial}


if __name__ == "__main__":
    train, val, test = preparar_particiones()
    ds_train, ds_val, ds_test = (construir_dataset(d) for d in (train, val, test))
    y_val, y_test = val["etiquetas"].tolist(), test["etiquetas"].tolist()
    CARPETA_MODELOS.mkdir(exist_ok=True)

    corridas = [entrenar(s, ds_train, ds_val, y_val) for s in CONFIG["semillas"]]
    resumen = pd.DataFrame([{k: v for k, v in r.items() if k != "historial"} for r in corridas])
    print(resumen.to_string(index=False))
    print("F1 val: media =", round(resumen["mejor_f1_val"].mean(), 4),
          "desviación =", round(resumen["mejor_f1_val"].std(ddof=0), 4))

    # Selección SOLO por validación; test se usa una vez al final.
    mejor = max(corridas, key=lambda r: r["mejor_f1_val"])
    modelo = ModeloBETOCRF().to(DISPOSITIVO)
    modelo.load_state_dict(torch.load(mejor["checkpoint"], map_location=DISPOSITIVO, weights_only=True))
    rep_val = metricas_estrictas(y_val, predecir(modelo, ds_val))
    rep_test = metricas_estrictas(y_test, predecir(modelo, ds_test))
    print("Mejor semilla:", mejor["semilla"])
    print("VALIDACIÓN APF3\n", rep_val.round(4).to_string(index=False))
    print("TEST APF3\n", rep_test.round(4).to_string(index=False))

    destino = CARPETA_MODELOS / "apf3_beto_crf.pt"
    torch.save(modelo.state_dict(), destino)
    for r in corridas:  # borrar checkpoints intermedios (~440 MB cada uno)
        CARPETA_MODELOS.joinpath(f"apf3_semilla_{r['semilla']}.pt").unlink(missing_ok=True)

    with open(CARPETA_MODELOS / "metricas_apf3.json", "w", encoding="utf-8") as f:
        json.dump({"mejor_semilla": mejor["semilla"], "config": CONFIG,
                   "semillas": resumen.to_dict("records"),
                   "validacion": rep_val.to_dict("records"), "test": rep_test.to_dict("records"),
                   "historial": [h for r in corridas for h in r["historial"]]},
                  f, ensure_ascii=False, indent=2)
    print("Modelo guardado en", destino)
