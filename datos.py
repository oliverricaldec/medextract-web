"""
Preparación de datos y métricas, idéntica al notebook (semilla 42):
limpieza BIOES -> sin conflictos -> sin duplicados -> piloto estratificado de 3000 -> 60/20/20.
"""

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from medextract import MAPA_ETIQUETAS

SEMILLA = 42
MAX_TOKENS_ORACION = 128
TAMANO_PILOTO = 3000
TIPOS_ENTIDAD = ["PROBLEM", "TREATMENT", "TEST"]


def separar(etiqueta):
    return ("O", None) if etiqueta == "O" else etiqueta.split("-", 1)


def bioes_valida(secuencia):
    abierta = None
    for etiqueta in secuencia:
        prefijo, tipo = separar(etiqueta)
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


def _firma(etiquetas, longitud):
    tipos = sorted({e.split("-", 1)[1] for e in etiquetas if e != "O"})
    largo = "CORTA" if longitud <= 10 else "MEDIA" if longitud <= 30 else "LARGA"
    return ("+".join(tipos) or "SIN_ENTIDAD") + "_" + largo


def _agrupar_raros(df, minimo=10):
    df = df.copy()
    conteos = df["estrato"].value_counts()
    df.loc[df["estrato"].isin(conteos[conteos < minimo].index), "estrato"] = "OTROS"
    return df


def preparar_particiones():
    print("Descargando dataset HUMADEX/spanish_ner_dataset ...")
    df = load_dataset("HUMADEX/spanish_ner_dataset", split="train").to_pandas()
    df["tokens"] = df["sentence"].apply(list)
    df["tags_numericos"] = df["tags"].apply(lambda v: [int(x) for x in v])
    df["etiquetas"] = df["tags_numericos"].apply(lambda v: [MAPA_ETIQUETAS[x] for x in v])
    df = df[df["tokens"].str.len() == df["etiquetas"].str.len()]

    df["clave_tokens"] = df["tokens"].apply(lambda t: "␟".join(map(str, t)))
    df["clave_etiquetas"] = df["etiquetas"].apply("␟".join)
    conflictos = df.groupby("clave_tokens")["clave_etiquetas"].nunique()
    conflictos = set(conflictos[conflictos > 1].index)
    df = df[df["etiquetas"].apply(bioes_valida) & ~df["clave_tokens"].isin(conflictos)]
    df = df.drop_duplicates(subset=["clave_tokens", "clave_etiquetas"]).reset_index(drop=True)

    df["longitud"] = df["tokens"].str.len()
    df = df[(df["longitud"] > 0) & (df["longitud"] <= MAX_TOKENS_ORACION)].copy()
    df["estrato"] = [_firma(e, l) for e, l in zip(df["etiquetas"], df["longitud"])]
    df = _agrupar_raros(df)
    piloto, _ = train_test_split(df, train_size=TAMANO_PILOTO, random_state=SEMILLA, stratify=df["estrato"])
    piloto = _agrupar_raros(piloto.sample(frac=1, random_state=SEMILLA).reset_index(drop=True))

    train, temporal = train_test_split(piloto, test_size=0.40, random_state=SEMILLA, stratify=piloto["estrato"])
    temporal = _agrupar_raros(temporal, minimo=4)
    val, test = train_test_split(temporal, test_size=0.50, random_state=SEMILLA, stratify=temporal["estrato"])
    train, val, test = (d.reset_index(drop=True) for d in (train, val, test))
    print("Train:", len(train), "| Val:", len(val), "| Test:", len(test))
    return train, val, test


def extraer_entidades(secuencia):
    """Entidades estrictas (misma función del notebook): solo cuenta B..E completos y S."""
    entidades, i = set(), 0
    while i < len(secuencia):
        prefijo, tipo = separar(secuencia[i])
        if prefijo == "S":
            entidades.add((tipo, i, i))
            i += 1
            continue
        if prefijo == "B":
            inicio, sig = i, i + 1
            while sig < len(secuencia):
                p2, t2 = separar(secuencia[sig])
                if t2 != tipo:
                    break
                if p2 == "I":
                    sig += 1
                    continue
                if p2 == "E":
                    entidades.add((tipo, inicio, sig))
                break
            i = max(i + 1, sig + 1)
            continue
        i += 1
    return entidades


def metricas_estrictas(y_real, y_pred):
    reales, predichas = set(), set()
    for n, (r, p) in enumerate(zip(y_real, y_pred)):
        reales.update((n,) + e for e in extraer_entidades(r))
        predichas.update((n,) + e for e in extraer_entidades(p))
    filas = []
    for tipo in TIPOS_ENTIDAD + ["MICRO"]:
        R = reales if tipo == "MICRO" else {e for e in reales if e[1] == tipo}
        P = predichas if tipo == "MICRO" else {e for e in predichas if e[1] == tipo}
        tp, fp, fn = len(R & P), len(P - R), len(R - P)
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        filas.append({"entidad": tipo, "precision": prec, "recall": rec, "f1": f1, "TP": tp, "FP": fp, "FN": fn})
    return pd.DataFrame(filas)
