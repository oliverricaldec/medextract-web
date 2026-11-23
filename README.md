# MedExtract Web

Interfaz web para los modelos NER médicos del notebook `MedExtract_APF1_APF2_APF3_BETO_CRF.ipynb`.
Detecta en textos clínicos en español tres tipos de entidad: **PROBLEM**, **TREATMENT** y **TEST**.

Funciones:
- Resaltado de entidades en el texto, por colores.
- Lista de entidades agrupadas por tipo y conteo.
- Comparación lado a lado de APF1 / APF2 / APF3 (marca varios modelos).
- Etiqueta BIOES por palabra.
- Descarga de resultados en CSV o JSON.
- Tabla con las métricas de test obtenidas en el notebook.

## Estructura

```
app.py                   servidor Flask (API + página)
medextract.py            carga de modelos e inferencia (mismo código del notebook)
static/index.html        interfaz
entrenar_apf1_local.py   entrena APF1 en tu PC (sin Colab)
exportar_desde_colab.py  celda para exportar los modelos desde Colab
modelos/                 aquí van los modelos
```

## Puesta en marcha

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Opción rápida: solo APF1 (CPU, ~2 min)

```powershell
python entrenar_apf1_local.py
python app.py
```

Abre http://localhost:5000

### Opción completa: APF1 + APF2 + APF3 desde Colab

1. En Colab, después de entrenar, pega y ejecuta el contenido de `exportar_desde_colab.py` en una celda nueva.
2. Descomprime `modelos_medextract.zip` dentro de `modelos/`:
   ```
   modelos/crf_rasgos_manuales.joblib
   modelos/apf2_beto_finetuned/   (config.json, model.safetensors, tokenizer...)
   modelos/apf3_beto_crf.pt
   ```
3. Instala también `torch` y `transformers` (`pip install torch transformers`).
4. `python app.py`

La web solo carga los modelos que encuentre; si falta alguno, simplemente no aparece.
APF2/APF3 funcionan en CPU (≈0.1–0.5 s por oración).

## API

`POST /api/analizar`

```json
{ "texto": "El paciente presenta diabetes...", "modelos": ["apf1", "apf2"] }
```

Respuesta: por cada modelo, `tokens` (palabra, posición, etiqueta BIOES) y `entidades` (tipo, texto, inicio, fin).

`GET /api/modelos` lista los modelos cargados.

## Notas

- El texto se separa por espacios (igual que `texto.split()` en el notebook) y se divide en oraciones
  por `. ! ? ;` porque los modelos se entrenaron con oraciones cortas.
- Si el modelo produce una secuencia BIOES inválida (p. ej. `I-` sin `B-`), la web igual agrupa la entidad
  para no perderla.
- Proyecto académico: no usar para decisiones médicas reales.
