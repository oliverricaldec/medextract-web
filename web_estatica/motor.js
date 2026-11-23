/*
 * MedExtract — motor de inferencia en el navegador.
 * Reproduce medextract.py: mismo corte en palabras y oraciones, tokenizador WordPiece de BETO,
 * BETO (ONNX) + Viterbi del CRF con reglas BIOES (APF3) y CRF de rasgos manuales (APF1).
 */
(function () {
  const MAX_SUBTOKENS = 128;
  const MAX_PALABRAS_ORACION = 60;
  const LOCAL = ["localhost", "127.0.0.1"].includes(location.hostname);
  const BASE = LOCAL ? "modelo/" : "https://huggingface.co/oliverj1411/medextract-modelos/resolve/main/";

  // ---------------- texto -> palabras -> oraciones (igual que medextract.py) ----------------
  function tokenizar(texto) {
    return [...texto.matchAll(/\S+/g)].map((m) => [m[0], m.index, m.index + m[0].length]);
  }

  function partirOraciones(palabras) {
    const oraciones = [];
    let actual = [];
    for (const p of palabras) {
      actual.push(p);
      if (".!?;".includes(p[0].at(-1)) || actual.length >= MAX_PALABRAS_ORACION) {
        oraciones.push(actual);
        actual = [];
      }
    }
    if (actual.length) oraciones.push(actual);
    return oraciones;
  }

  function agruparEntidades(palabras, etiquetas, texto) {
    const entidades = [];
    let abierta = null;
    const cerrar = () => {
      if (abierta) { abierta.texto = texto.slice(abierta.inicio, abierta.fin); entidades.push(abierta); }
      abierta = null;
    };
    palabras.forEach(([, ini, fin], i) => {
      const etq = etiquetas[i];
      if (etq === "O") return cerrar();
      const [prefijo, tipo] = etq.split("-");
      const continua = abierta && abierta.tipo === tipo && (prefijo === "I" || prefijo === "E");
      if (!continua) { cerrar(); abierta = { tipo, inicio: ini, fin }; }
      abierta.fin = fin;
      if (prefijo === "E" || prefijo === "S") cerrar();
    });
    cerrar();
    return entidades;
  }

  // ---------------- Tokenizador BERT (BertNormalizer cased + BertPreTokenizer + WordPiece) ----------------
  let vocab = null;
  const esPuntuacion = (c) => {
    const cp = c.codePointAt(0);
    return (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) ||
           (cp >= 123 && cp <= 126) || /\p{P}/u.test(c);
  };
  const esCJK = (cp) => (cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0x3400 && cp <= 0x4dbf) ||
    (cp >= 0x20000 && cp <= 0x2a6df) || (cp >= 0x2a700 && cp <= 0x2b81f) || (cp >= 0x2b820 && cp <= 0x2ceaf) ||
    (cp >= 0xf900 && cp <= 0xfaff) || (cp >= 0x2f800 && cp <= 0x2fa1f);

  function preTokenizar(palabra) {
    const piezas = [];
    let actual = "";
    for (const c of palabra) {
      const cp = c.codePointAt(0);
      if (cp === 0 || cp === 0xfffd || /\p{Cc}|\p{Cf}/u.test(c) && !/[\t\n\r]/.test(c)) continue; // clean_text
      if (/\s/.test(c)) { if (actual) piezas.push(actual); actual = ""; continue; }
      if (esPuntuacion(c) || esCJK(cp)) {
        if (actual) piezas.push(actual);
        piezas.push(c);
        actual = "";
      } else actual += c;
    }
    if (actual) piezas.push(actual);
    return piezas;
  }

  function wordPiece(pieza) {
    const chars = [...pieza];
    if (chars.length > 100) return [vocab.get("[UNK]")];
    const ids = [];
    let inicio = 0;
    while (inicio < chars.length) {
      let fin = chars.length, id;
      while (inicio < fin) {
        const sub = (inicio > 0 ? "##" : "") + chars.slice(inicio, fin).join("");
        if (vocab.has(sub)) { id = vocab.get(sub); break; }
        fin--;
      }
      if (id === undefined) return [vocab.get("[UNK]")];
      ids.push(id);
      inicio = fin;
    }
    return ids;
  }

  function idsPalabra(palabra) {
    const ids = preTokenizar(palabra).flatMap(wordPiece);
    return ids.length ? ids : [vocab.get("[UNK]")];
  }

  // ---------------- APF3: BETO (ONNX) + CRF con reglas BIOES ----------------
  let sesion = null, crf3 = null;

  async function predecirAPF3(oracion) {
    const ids = [vocab.get("[CLS]")], primeras = [];
    for (const palabra of oracion) {
      const piezas = idsPalabra(palabra);
      if (ids.length + piezas.length > MAX_SUBTOKENS - 1) break; // truncado como en Python
      primeras.push(ids.length);
      ids.push(...piezas);
    }
    ids.push(vocab.get("[SEP]"));
    const n = ids.length;
    const salida = await sesion.run({
      input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, n]),
      attention_mask: new ort.Tensor("int64", new BigInt64Array(n).fill(1n), [1, n]),
    });
    const em = salida.emisiones.data, T = crf3.etiquetas.length;
    const emisiones = primeras.map((pos) => Array.from(em.subarray(pos * T, pos * T + T)));
    const ruta = viterbi(emisiones, crf3.inicio, crf3.fin, crf3.trans);
    const etiquetas = ruta.map((i) => crf3.etiquetas[i]);
    while (etiquetas.length < oracion.length) etiquetas.push("O");
    return etiquetas;
  }

  function viterbi(em, inicio, fin, trans) {
    if (!em.length) return [];
    const T = em[0].length;
    let dp = em[0].map((v, j) => v + (inicio ? inicio[j] : 0));
    const historial = [];
    for (let t = 1; t < em.length; t++) {
      const nuevo = new Array(T), origen = new Array(T);
      for (let j = 0; j < T; j++) {
        let mejor = -Infinity, arg = 0;
        for (let i = 0; i < T; i++) {
          const v = dp[i] + trans[i][j];
          if (v > mejor) { mejor = v; arg = i; }
        }
        nuevo[j] = mejor + em[t][j];
        origen[j] = arg;
      }
      dp = nuevo;
      historial.push(origen);
    }
    let ultimo = 0, mejor = -Infinity;
    dp.forEach((v, j) => { const s = v + (fin ? fin[j] : 0); if (s > mejor) { mejor = s; ultimo = j; } });
    const ruta = [ultimo];
    for (let t = historial.length - 1; t >= 0; t--) { ultimo = historial[t][ultimo]; ruta.push(ultimo); }
    return ruta.reverse();
  }

  // ---------------- APF1: CRF + rasgos manuales (sklearn-crfsuite) ----------------
  let crf1 = null;

  function rasgosToken(oracion, i) {
    const p = oracion[i], ch = [...p];
    const r = {
      bias: 1.0,
      "palabra.lower": p.toLowerCase(),
      prefijo2: ch.slice(0, 2).join("").toLowerCase(),
      sufijo2: ch.slice(-2).join("").toLowerCase(),
      sufijo3: ch.slice(-3).join("").toLowerCase(),
      es_mayuscula: /\p{Lu}/u.test(p) && !/\p{Ll}/u.test(p),
      inicia_mayuscula: esTitulo(p),
      es_numero: /^\p{Nd}+$/u.test(p),
      contiene_numero: /\p{Nd}/u.test(p),
      BOS: i === 0,
      EOS: i === oracion.length - 1,
    };
    if (i > 0) r["anterior.lower"] = oracion[i - 1].toLowerCase();
    if (i < oracion.length - 1) r["siguiente.lower"] = oracion[i + 1].toLowerCase();
    return r;
  }

  // str.istitle() de Python: mayúscula solo tras no-letra, minúscula solo tras letra, al menos una letra con caso
  function esTitulo(p) {
    let previaLetra = false, alguna = false;
    for (const c of p) {
      const may = /\p{Lu}|\p{Lt}/u.test(c), min = /\p{Ll}/u.test(c);
      if (may) { if (previaLetra) return false; previaLetra = true; alguna = true; }
      else if (min) { if (!previaLetra) return false; previaLetra = true; alguna = true; }
      else previaLetra = false;
    }
    return alguna;
  }

  function predecirAPF1(oracion) {
    const E = crf1.etiquetas;
    const em = oracion.map((_, i) => {
      const puntaje = new Array(E.length).fill(0);
      for (const [nombre, valor] of Object.entries(rasgosToken(oracion, i))) {
        const [clave, peso] = typeof valor === "string" ? [`${nombre}:${valor}`, 1] : [nombre, Number(valor)];
        const pesos = crf1.estado[clave];
        if (pesos && peso) E.forEach((e, k) => { if (pesos[e]) puntaje[k] += pesos[e] * peso; });
      }
      return puntaje;
    });
    const trans = E.map((a) => E.map((b) => (crf1.trans[a] && crf1.trans[a][b]) || 0));
    return viterbi(em, null, null, trans).map((k) => E[k]);
  }

  // ---------------- carga y API pública ----------------
  async function descargar(nombre, onProgreso) {
    let cache = null;
    try { cache = await caches.open("medextract-v1"); } catch {}
    const url = BASE + nombre;
    let resp = cache && (await cache.match(url));
    if (resp) { onProgreso && onProgreso(1, 1); return resp; }
    resp = await fetch(url);
    if (!resp.ok) throw new Error(`No se pudo descargar ${nombre} (${resp.status})`);
    const total = +resp.headers.get("content-length") || 0;
    const lector = resp.body.getReader(), partes = [];
    let recibido = 0;
    for (;;) {
      const { done, value } = await lector.read();
      if (done) break;
      partes.push(value);
      recibido += value.length;
      onProgreso && onProgreso(recibido, total);
    }
    const blob = new Blob(partes);
    try { cache && (await cache.put(url, new Response(blob))); } catch {}
    return new Response(blob);
  }

  async function cargar(onProgreso) {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
    const [v, c3, c1] = await Promise.all([
      descargar("vocab.txt").then((r) => r.text()),
      descargar("apf3_crf.json").then((r) => r.json()),
      descargar("apf1.json").then((r) => r.json()),
    ]);
    vocab = new Map(v.split(/\r?\n/).map((p, i) => [p, i]));
    crf3 = c3;
    crf1 = c1;
    const onnx = await (await descargar("apf3.onnx", onProgreso)).arrayBuffer();
    sesion = await ort.InferenceSession.create(onnx, { executionProviders: ["wasm"] });
  }

  const MODELOS = {
    apf1: { nombre: "APF1 · CRF + rasgos manuales", predecir: async (o) => predecirAPF1(o) },
    apf3: { nombre: "APF3 · BETO + CRF neuronal (recomendado)", predecir: predecirAPF3 },
  };

  async function analizar(texto, ids) {
    const palabras = tokenizar(texto), oraciones = partirOraciones(palabras);
    const resultados = [];
    for (const id of ids) {
      const t0 = performance.now(), etiquetas = [];
      for (const o of oraciones) etiquetas.push(...(await MODELOS[id].predecir(o.map((p) => p[0]))));
      resultados.push({
        modelo: id,
        nombre: MODELOS[id].nombre,
        tokens: palabras.map(([texto, inicio, fin], i) => ({ texto, inicio, fin, etiqueta: etiquetas[i] })),
        entidades: agruparEntidades(palabras, etiquetas, texto),
        tiempo_ms: Math.round((performance.now() - t0) * 10) / 10,
      });
    }
    return { texto, resultados };
  }

  window.MedExtract = {
    cargar, analizar,
    modelos: Object.entries(MODELOS).map(([id, m]) => ({ id, nombre: m.nombre })),
    _idsPalabra: (p) => idsPalabra(p), // para pruebas
  };
})();
