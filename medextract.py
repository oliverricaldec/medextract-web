"""
MedExtract — carga de modelos e inferencia.

Reproduce la misma lógica del notebook (APF1, APF2, APF3) para usarla fuera de Colab.
Los modelos se buscan en la carpeta ./modelos :

    modelos/
      crf_rasgos_manuales.joblib      -> APF1 (CRF + rasgos manuales)
      apf2_beto_finetuned/            -> APF2 (BETO fine-tuned, save_pretrained)
      apf3_beto_crf.pt                -> APF3 (BETO + CRF neuronal, state_dict)

Solo se cargan los que existan. torch/transformers se importan solo si hace falta.
"""

import re
import time
from pathlib import Path

CARPETA_MODELOS = Path(__file__).parent / "modelos"
NOMBRE_BETO = "dccuchile/bert-base-spanish-wwm-cased"
MAX_SUBTOKENS = 128
MAX_PALABRAS_ORACION = 60

MAPA_ETIQUETAS = {
    0: "O",
    1: "B-PROBLEM", 2: "I-PROBLEM", 3: "E-PROBLEM", 4: "S-PROBLEM",
    5: "B-TREATMENT", 6: "I-TREATMENT", 7: "E-TREATMENT", 8: "S-TREATMENT",
    9: "B-TEST", 10: "I-TEST", 11: "E-TEST", 12: "S-TEST",
}


# ---------------------------------------------------------------------------
# Texto -> palabras -> oraciones
# ---------------------------------------------------------------------------

def tokenizar(texto):
    """Igual que texto.split() del notebook, pero guardando la posición de cada palabra."""
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", texto)]


def partir_oraciones(palabras):
    """Agrupa palabras en oraciones (los modelos se entrenaron con oraciones cortas)."""
    oraciones, actual = [], []
    for palabra in palabras:
        actual.append(palabra)
        fin_oracion = palabra[0][-1] in ".!?;"
        if fin_oracion or len(actual) >= MAX_PALABRAS_ORACION:
            oraciones.append(actual)
            actual = []
    if actual:
        oraciones.append(actual)
    return oraciones


def agrupar_entidades(palabras, etiquetas, texto):
    """
    Convierte etiquetas BIOES en entidades. Es tolerante: si el modelo produce una
    secuencia inválida (p. ej. I sin B) igualmente arma la entidad, para no perderla.
    """
    entidades, abierta = [], None

    def cerrar():
        nonlocal abierta
        if abierta:
            abierta["texto"] = texto[abierta["inicio"]:abierta["fin"]]
            entidades.append(abierta)
        abierta = None

    for (_, ini, fin), etiqueta in zip(palabras, etiquetas):
        if etiqueta == "O":
            cerrar()
            continue
        prefijo, tipo = etiqueta.split("-", 1)
        continua = abierta is not None and abierta["tipo"] == tipo and prefijo in ("I", "E")
        if not continua:
            cerrar()
            abierta = {"tipo": tipo, "inicio": ini, "fin": fin}
        abierta["fin"] = fin
        if prefijo in ("E", "S"):
            cerrar()
    cerrar()
    return entidades


# ---------------------------------------------------------------------------
# APF1 — CRF con rasgos manuales (copiado del notebook)
# ---------------------------------------------------------------------------

def rasgos_token(oracion, indice):
    palabra = str(oracion[indice])
    rasgos = {
        "bias": 1.0,
        "palabra.lower": palabra.lower(),
        "prefijo2": palabra[:2].lower(),
        "sufijo2": palabra[-2:].lower(),
        "sufijo3": palabra[-3:].lower(),
        "es_mayuscula": palabra.isupper(),
        "inicia_mayuscula": palabra.istitle(),
        "es_numero": palabra.isdigit(),
        "contiene_numero": any(c.isdigit() for c in palabra),
        "BOS": indice == 0,
        "EOS": indice == len(oracion) - 1,
    }
    if indice > 0:
        rasgos["anterior.lower"] = str(oracion[indice - 1]).lower()
    if indice < len(oracion) - 1:
        rasgos["siguiente.lower"] = str(oracion[indice + 1]).lower()
    return rasgos


def rasgos_oracion(oracion):
    return [rasgos_token(oracion, i) for i in range(len(oracion))]


class APF1:
    id = "apf1"
    nombre = "APF1 · CRF + rasgos manuales"

    def __init__(self, ruta):
        import joblib
        self.crf = joblib.load(ruta)

    def predecir(self, oraciones):
        return self.crf.predict([rasgos_oracion(o) for o in oraciones])


# ---------------------------------------------------------------------------
# APF2 — BETO fine-tuned (token classification)
# ---------------------------------------------------------------------------

def _primeras_subpiezas(word_ids):
    """Posición del primer sub-token de cada palabra (misma alineación del notebook)."""
    posiciones, previo = {}, None
    for pos, w in enumerate(word_ids):
        if w is not None and w != previo:
            posiciones[w] = pos
        previo = w
    return posiciones


class APF2:
    id = "apf2"
    nombre = "APF2 · BETO fine-tuned"

    def __init__(self, ruta):
        import torch
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        self.torch = torch
        self.tokenizador = AutoTokenizer.from_pretrained(ruta, use_fast=True)
        self.modelo = AutoModelForTokenClassification.from_pretrained(ruta).eval()

    def predecir(self, oraciones):
        resultado = []
        for oracion in oraciones:
            entrada = self.tokenizador(oracion, is_split_into_words=True, truncation=True,
                                       max_length=MAX_SUBTOKENS, return_tensors="pt")
            with self.torch.no_grad():
                pred = self.modelo(**entrada).logits.argmax(-1)[0].tolist()
            posiciones = _primeras_subpiezas(entrada.word_ids(0))
            resultado.append([
                MAPA_ETIQUETAS[pred[posiciones[i]]] if i in posiciones else "O"
                for i in range(len(oracion))
            ])
        return resultado


# ---------------------------------------------------------------------------
# APF3 — BETO + CRF neuronal con transiciones BIOES restringidas
# ---------------------------------------------------------------------------

def _construir_modelo_apf3():
    """Mismas clases del notebook (APF3CRF + ModeloBETOCRF), solo la parte de inferencia."""
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel

    def reglas_bioes(id2label):
        n = len(id2label)
        inicio = torch.zeros(n, dtype=torch.bool)
        fin = torch.zeros(n, dtype=torch.bool)
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
            self.trans = nn.Parameter(torch.zeros(n_tags, n_tags))
            inicio, fin, pasos = reglas_bioes(id2label)
            self.register_buffer("inicio_valido", inicio)
            self.register_buffer("fin_valido", fin)
            self.register_buffer("trans_valida", pasos)

        def puntajes(self):
            return (self.inicio.masked_fill(~self.inicio_valido, -1e4),
                    self.fin.masked_fill(~self.fin_valido, -1e4),
                    self.trans.masked_fill(~self.trans_valida, -1e4))

        @torch.no_grad()
        def decode(self, em):
            """Viterbi para una sola oración: em [palabras, tags]."""
            ini, fin, trans = self.puntajes()
            dp, historial = ini + em[0], []
            for pos in range(1, em.shape[0]):
                mejor, origen = (dp.unsqueeze(1) + trans).max(dim=0)
                dp = mejor + em[pos]
                historial.append(origen)
            ultimo = int((dp + fin).argmax())
            ruta = [ultimo]
            for origen in reversed(historial):
                ultimo = int(origen[ultimo])
                ruta.append(ultimo)
            return list(reversed(ruta))

    class ModeloBETOCRF(nn.Module):
        def __init__(self):
            super().__init__()
            # from_config: no descarga los pesos base, se sobrescriben con el checkpoint
            self.beto = AutoModel.from_config(AutoConfig.from_pretrained(NOMBRE_BETO))
            self.dropout = nn.Dropout(0.1)
            self.clasificador = nn.Linear(self.beto.config.hidden_size, len(MAPA_ETIQUETAS))
            self.crf = APF3CRF(len(MAPA_ETIQUETAS), MAPA_ETIQUETAS)

    return ModeloBETOCRF()


class APF3:
    id = "apf3"
    nombre = "APF3 · BETO + CRF neuronal"

    def __init__(self, ruta):
        import torch
        from transformers import AutoTokenizer
        self.torch = torch
        self.tokenizador = AutoTokenizer.from_pretrained(NOMBRE_BETO, use_fast=True)
        self.modelo = _construir_modelo_apf3()
        estado = torch.load(ruta, map_location="cpu", weights_only=True)
        faltantes, _ = self.modelo.load_state_dict(estado, strict=False)
        faltantes = [k for k in faltantes if "position_ids" not in k]
        if faltantes:
            raise ValueError(f"El checkpoint no coincide con la arquitectura APF3: {faltantes[:5]}")
        self.modelo.eval()

    def predecir(self, oraciones):
        resultado = []
        for oracion in oraciones:
            entrada = self.tokenizador(oracion, is_split_into_words=True, truncation=True,
                                       max_length=MAX_SUBTOKENS, return_tensors="pt")
            posiciones = _primeras_subpiezas(entrada.word_ids(0))
            with self.torch.no_grad():
                ocultos = self.modelo.beto(**entrada).last_hidden_state[0]
                indices = [posiciones[i] for i in sorted(posiciones)]
                emisiones = self.modelo.clasificador(ocultos[indices])
                ids = self.modelo.crf.decode(emisiones)
            etiquetas = [MAPA_ETIQUETAS[i] for i in ids]
            resultado.append(etiquetas + ["O"] * (len(oracion) - len(etiquetas)))
        return resultado


# ---------------------------------------------------------------------------
# Registro de modelos disponibles
# ---------------------------------------------------------------------------

ARCHIVOS = [
    (APF1, "crf_rasgos_manuales.joblib"),
    (APF2, "apf2_beto_finetuned"),
    (APF3, "apf3_beto_crf.pt"),
]


def cargar_modelos():
    modelos, errores = {}, {}
    for clase, archivo in ARCHIVOS:
        ruta = CARPETA_MODELOS / archivo
        if not ruta.exists():
            continue
        try:
            print(f"Cargando {clase.nombre} desde {ruta} ...")
            modelos[clase.id] = clase(str(ruta))
        except Exception as e:  # p. ej. falta torch instalado
            errores[clase.id] = f"{type(e).__name__}: {e}"
            print(f"  No se pudo cargar {clase.id}: {errores[clase.id]}")
    return modelos, errores


def analizar(modelo, texto):
    inicio = time.perf_counter()
    palabras = tokenizar(texto)
    oraciones = partir_oraciones(palabras)
    etiquetas = []
    if oraciones:
        for etiquetas_oracion in modelo.predecir([[p[0] for p in o] for o in oraciones]):
            etiquetas.extend(etiquetas_oracion)
    return {
        "modelo": modelo.id,
        "nombre": modelo.nombre,
        "tokens": [
            {"texto": p, "inicio": i, "fin": f, "etiqueta": e}
            for (p, i, f), e in zip(palabras, etiquetas)
        ],
        "entidades": agrupar_entidades(palabras, etiquetas, texto),
        "tiempo_ms": round((time.perf_counter() - inicio) * 1000, 1),
    }
