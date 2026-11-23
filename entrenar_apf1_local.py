"""
Entrena APF1 (CRF + rasgos manuales) en tu PC, sin Colab ni GPU (~1-2 min).
Reproduce el mismo preprocesamiento y partición del notebook (semilla 42).

    python entrenar_apf1_local.py
"""

import joblib
import pandas as pd
import sklearn_crfsuite
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from medextract import CARPETA_MODELOS, MAPA_ETIQUETAS, rasgos_oracion

SEMILLA = 42
MAX_TOKENS_ORACION = 128
TAMANO_PILOTO = 3000


def bioes_valida(secuencia):
    abierta = None
    for etiqueta in secuencia:
        prefijo, tipo = ("O", None) if etiqueta == "O" else etiqueta.split("-", 1)
        if prefijo in {"O", "S"}:
            if abierta is not None:
                return False
        elif prefijo == "B":
            if abierta is not None:
                return False
            abierta = tipo
        elif prefijo in {"I", "E"}:
            if abierta != tipo:
                return False
            if prefijo == "E":
                abierta = None
    return abierta is None


def firma(etiquetas, longitud):
    tipos = sorted({e.split("-", 1)[1] for e in etiquetas if e != "O"})
    largo = "CORTA" if longitud <= 10 else "MEDIA" if longitud <= 30 else "LARGA"
    return ("+".join(tipos) or "SIN_ENTIDAD") + "_" + largo


def agrupar_raros(df, minimo=10):
    df = df.copy()
    conteos = df["estrato"].value_counts()
    df.loc[df["estrato"].isin(conteos[conteos < minimo].index), "estrato"] = "OTROS"
    return df


print("Descargando dataset HUMADEX/spanish_ner_dataset ...")
df = load_dataset("HUMADEX/spanish_ner_dataset", split="train").to_pandas()
df["tokens"] = df["sentence"].apply(list)
df["etiquetas"] = df["tags"].apply(lambda v: [MAPA_ETIQUETAS[int(x)] for x in v])
df = df[df["tokens"].str.len() == df["etiquetas"].str.len()]

# Limpieza: BIOES válidas, sin anotaciones contradictorias, sin duplicados
df["clave_tokens"] = df["tokens"].apply(lambda t: "␟".join(map(str, t)))
df["clave_etiquetas"] = df["etiquetas"].apply("␟".join)
conflictos = df.groupby("clave_tokens")["clave_etiquetas"].nunique()
conflictos = set(conflictos[conflictos > 1].index)
df = df[df["etiquetas"].apply(bioes_valida) & ~df["clave_tokens"].isin(conflictos)]
df = df.drop_duplicates(subset=["clave_tokens", "clave_etiquetas"]).reset_index(drop=True)

# Piloto estratificado
df["longitud"] = df["tokens"].str.len()
df = df[(df["longitud"] > 0) & (df["longitud"] <= MAX_TOKENS_ORACION)].copy()
df["estrato"] = [firma(e, l) for e, l in zip(df["etiquetas"], df["longitud"])]
df = agrupar_raros(df)
piloto, _ = train_test_split(df, train_size=TAMANO_PILOTO, random_state=SEMILLA, stratify=df["estrato"])
piloto = agrupar_raros(piloto.sample(frac=1, random_state=SEMILLA).reset_index(drop=True))

# 60 / 20 / 20
train, temporal = train_test_split(piloto, test_size=0.40, random_state=SEMILLA, stratify=piloto["estrato"])
temporal = agrupar_raros(temporal, minimo=4)
val, test = train_test_split(temporal, test_size=0.50, random_state=SEMILLA, stratify=temporal["estrato"])
print("Train:", len(train), "| Val:", len(val), "| Test:", len(test))

crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100,
                           all_possible_transitions=True)
crf.fit([rasgos_oracion(o) for o in train["tokens"]], train["etiquetas"].tolist())

# F1 por palabra-etiqueta en test (rápido, solo como comprobación)
pred = crf.predict([rasgos_oracion(o) for o in test["tokens"]])
reales = [e for s in test["etiquetas"] for e in s]
predichas = [e for s in pred for e in s]
aciertos = sum(r == p != "O" for r, p in zip(reales, predichas))
print(f"Aciertos en etiquetas de entidad (test): {aciertos} / {sum(r != 'O' for r in reales)}")

CARPETA_MODELOS.mkdir(exist_ok=True)
destino = CARPETA_MODELOS / "crf_rasgos_manuales.joblib"
joblib.dump(crf, destino)
print("Modelo guardado en", destino)
