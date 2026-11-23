"""
MedExtract Web — servidor Flask.

    python app.py      ->  http://localhost:5000
"""

from flask import Flask, jsonify, request, send_from_directory

import medextract

MAX_CARACTERES = 5000

app = Flask(__name__, static_folder="static")
MODELOS, ERRORES = medextract.cargar_modelos()


@app.get("/")
def inicio():
    return send_from_directory("static", "index.html")


@app.get("/api/modelos")
def listar_modelos():
    return jsonify({
        "modelos": [{"id": m.id, "nombre": m.nombre} for m in MODELOS.values()],
        "errores": ERRORES,
    })


@app.post("/api/analizar")
def analizar():
    datos = request.get_json(silent=True) or {}
    texto = (datos.get("texto") or "").strip()
    ids = datos.get("modelos") or list(MODELOS)

    if not texto:
        return jsonify({"error": "Escribe un texto para analizar."}), 400
    if len(texto) > MAX_CARACTERES:
        return jsonify({"error": f"El texto supera {MAX_CARACTERES} caracteres."}), 400
    if not MODELOS:
        return jsonify({"error": "No hay modelos cargados. Revisa la carpeta modelos/ (ver README)."}), 503

    desconocidos = [i for i in ids if i not in MODELOS]
    if desconocidos:
        return jsonify({"error": f"Modelo no disponible: {', '.join(desconocidos)}"}), 400

    return jsonify({
        "texto": texto,
        "resultados": [medextract.analizar(MODELOS[i], texto) for i in ids],
    })


if __name__ == "__main__":
    print("Modelos cargados:", ", ".join(MODELOS) or "ninguno")
    app.run(host="127.0.0.1", port=5000, debug=False)
