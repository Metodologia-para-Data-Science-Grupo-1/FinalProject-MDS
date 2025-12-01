from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pickle

from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore
import io
from datetime import datetime

MODEL_PATH = Path(__file__).with_name("kmeans.pkl")
PROFILES_PATH = Path(__file__).parent.parent / "notebook" / "cluster_profiles.json"


def load_model(path: Path) -> Any:
    if not path.exists():
        return None
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except pickle.UnpicklingError:
        with open(path, "rb") as f:
            return pickle.load(f, fix_imports=True, encoding="latin1")


model = load_model(MODEL_PATH)

app = FastAPI(title="Customer Segmentation (Friendly) KMeans API", version="1.0")


# --- Utilities copied/adapted from original service ---
def _detect_kmeans_component(obj: Any) -> Any:
    step = None
    named_steps = getattr(obj, "named_steps", None)
    if isinstance(named_steps, dict):
        for key in ("kmeans", "model", "estimator", "cluster", "clf", "final"):
            if key in named_steps and hasattr(named_steps[key], "cluster_centers_"):
                step = named_steps[key]
                break
        if step is None:
            for comp in named_steps.values():
                if hasattr(comp, "cluster_centers_"):
                    step = comp
                    break
    if step is None and hasattr(obj, "cluster_centers_"):
        step = obj
    return step


def _feature_order(obj: Any) -> Optional[List[str]]:
    for candidate in (obj, getattr(obj, "named_steps", None), _detect_kmeans_component(obj)):
        if candidate is None:
            continue
        cols = getattr(candidate, "feature_names_in_", None)
        if cols is not None:
            return list(cols)
    for attr in ("feature_order", "features", "columns", "input_features"):
        cols = getattr(obj, attr, None)
        if cols is not None and isinstance(cols, (list, tuple)):
            return list(cols)
    return None


def _predict_cluster(values: Dict[str, float]) -> Dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado. Asegúrate de exportarlo como kmeans_rfm_pipeline.pkl en deployment/.")

    cols = _feature_order(model) or [
        "Recency",
        "Frequency",
        "Monetary",
        "AvgOrderValue",
        "TotalQuantity",
        "CustomerLifespan",
    ]

    if pd is not None:
        X = pd.DataFrame([[values.get(c) for c in cols]], columns=cols)
    else:
        X = [[values.get(c) for c in cols]]

    try:
        pred = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir con el modelo: {e}")

    label = int(pred[0]) if hasattr(pred, "__iter__") else int(pred)

    km = _detect_kmeans_component(model)
    distances = None
    try:
        if km is not None and hasattr(km, "transform"):
            if hasattr(model, "transform"):
                d = model.transform(X)
            else:
                d = km.transform(X)
            distances = [float(x) for x in d[0]]
    except Exception:
        distances = None

    return {
        "cluster": label,
        "feature_order": cols,
        "distances": distances,
        "n_clusters": int(getattr(km, "n_clusters", 0)) if km is not None else None,
    }


# Friendly field descriptions to serve UI and API docs
FIELD_DOCS = {
    "Recency": {
        "label": "Recencia",
        "description": "Días desde la última compra (valores más bajos = más reciente)",
        "example": 10,
    },
    "Frequency": {"label": "Frecuencia", "description": "Número total de compras", "example": 5},
    "Monetary": {"label": "Valor total gastado", "description": "Suma monetaria de todas las compras", "example": 250.0},
    "AvgOrderValue": {"label": "Promedio por compra", "description": "Monetary / Frequency (si aplica)", "example": 50.0},
    "TotalQuantity": {"label": "Total unidades compradas", "description": "Cantidad total de unidades adquiridas por el cliente", "example": 12},
    "CustomerLifespan": {"label": "Tiempo de cliente (días)", "description": "Días entre primera y última compra", "example": 400},
}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"model_loaded": model is not None, "model_path": str(MODEL_PATH), "has_kmeans": bool(_detect_kmeans_component(model)) if model is not None else False}


@app.get("/describe_fields")
def describe_fields() -> Dict[str, Any]:
    """Devuelve la documentación que usa la UI: etiquetas, descripción y ejemplo para cada campo."""
    return FIELD_DOCS


@app.get("/", response_class=HTMLResponse)
def friendly_form():
    # A small, interactive single-page UI that uses fetch to call /predict
    style = """
    body {font-family: Arial; max-width: 820px; margin: auto; padding: 18px}
    label {display: block; margin-top: 12px; font-weight:600}
    .field {display:flex; gap:12px; align-items:center}
    input {flex:1; padding:8px}
    small {display:block; color:#555}
    button {margin-top:18px; background:#1976d2; color:#fff; border:0; padding:10px 16px; border-radius:6px}
    .result {margin-top:18px; padding:12px; border-radius:6px; background:#f6f9ff; border:1px solid #dbe9ff}
    .cluster-badge {display:inline-block; padding:6px 10px; border-radius:20px; background:#e8f5e9; margin-right:8px}
    .help {font-size:0.9em; color:#666}
    """

    # Build HTML form fields dynamically from FIELD_DOCS
    fields_html = []
    for key, meta in FIELD_DOCS.items():
        fields_html.append(f"""
        <div class='field'>
          <div style='flex:2'>
            <label for='{key}'>{meta['label']} <span class='help'>(campo: {key})</span></label>
            <input id='{key}' name='{key}' type='number' step='any' placeholder='ej. {meta['example']}' required />
            <small>{meta['description']}</small>
          </div>
        </div>
        """
        )

    fields_joined = "\n".join(fields_html)

    # Inline JS to POST JSON to /predict and show friendly description
    script = """
    async function predict() {
      const payload = {};
      for (const key of Object.keys(FIELD_DOCS_JS)) {
        const el = document.getElementById(key);
        const v = el.value;
        if (v === '') { alert('Por favor rellena: ' + key); return; }
        payload[key] = Number(v);
      }
      const res = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
      if (!res.ok) {
        const txt = await res.text();
        document.getElementById('result').innerText = 'Error: ' + txt;
        return;
      }
      const data = await res.json();
      // Friendly mapping (can be extended)
      const names = {
        0: 'Clientes Estandar',
        1: 'Inactivos de bajo valor',
        2: 'Inactivos de bajo valor',
        3: 'Clientes Valiosos'
      };
      const clusterName = names[data.cluster] || ('Cluster ' + data.cluster);
      let html = `<div><span class='cluster-badge'>${clusterName}</span><strong>Cluster ${data.cluster}</strong></div>`;
      html += `<div style='margin-top:8px'><strong>Orden de features:</strong> ${data.feature_order.join(', ')}</div>`;
      if (data.n_clusters) html += `<div><strong>Total clusters:</strong> ${data.n_clusters}</div>`;
      if (data.distances) {
        const d = data.distances.map((v,i)=> i+': '+v.toFixed(4)).join(', ');
        html += `<div style='margin-top:8px'><strong>Distancias:</strong> ${d}</div>`;
      }
      document.getElementById('result').innerHTML = html;
    }
    """

    # Pass FIELD_DOCS to JS as JSON
    import json
    field_docs_js = json.dumps(FIELD_DOCS)

    html = f"""
    <html>
    <head>
      <title>Segmentación de Clientes - Interfaz Amigable</title>
      <style>{style}</style>
    </head>
    <body>
      <h2>Segmentación de Clientes (Interfaz Mejorada)</h2>
      <p class='help'>Introduce valores reales del cliente. Los ejemplos son sugeridos.</p>
      <div>
        {fields_joined}
      </div>
      <div>
        <button onclick='predict()'>Predecir clúster (sin recargar)</button>
      </div>
            <p style='margin-top:12px'><a href='/upload'>Analizar periodo completo (subir CSV/XLSX)</a></p>
      <div id='result' class='result' aria-live='polite'></div>
      <script>
        const FIELD_DOCS_JS = {field_docs_js};
        {script}
      </script>
    </body>
    </html>
    """
    return HTMLResponse(html)


@app.post("/predict")
def predict_json(payload: Dict[str, float]):
    out = _predict_cluster(payload)
    # Add brief human-readable recommendation for each cluster (discoverable mapping)
    descriptions = {
        0: "Clientes de comportamiento medio — considerar campañas de fidelización.",
        1: "Clientes inactivos y de bajo valor — posible reactivación con descuentos.",
        2: "Clientes inactivos y de bajo valor — evaluar limpieza de base o campañas de re-engagement.",
        3: "Clientes valiosos — premiar con programas VIP y retención.",
    }
    out["cluster_description"] = descriptions.get(out["cluster"], "")
    # If cluster_profiles.json exists, attach brief profile for this cluster
    try:
        import json
        if PROFILES_PATH.exists():
            with open(PROFILES_PATH, "r", encoding="utf-8") as f:
                profiles = json.load(f)
            prof = profiles.get(str(out["cluster"])) or profiles.get(out["cluster"]) or None
            if prof is not None:
                out["cluster_profile"] = prof
    except Exception:
        pass
    return JSONResponse(out)


@app.get("/clusters/perfiles")
def cluster_profiles() -> Dict[str, Any]:
    if not PROFILES_PATH.exists():
        raise HTTPException(status_code=404, detail="Archivo de perfiles no encontrado. Genera cluster_profiles.json en notebook/.")
    try:
        import json
        with open(PROFILES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer perfiles: {e}")


@app.post("/analyze_upload")
def analyze_upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Analiza un archivo CSV o XLSX de transacciones y devuelve el porcentaje de clientes por segmento.

    Formato esperado (columnas mínimas):
    - CustomerID: identificador del cliente
    - InvoiceDate: fecha de la transacción (YYYY-MM-DD o similar)
    - Quantity: unidades vendidas
    - UnitPrice: precio por unidad

    El endpoint agrupa por cliente dentro del archivo y calcula Recency, Frequency, Monetary,
    AvgOrderValue, TotalQuantity y CustomerLifespan sobre el periodo contenido en el archivo,
    luego aplica el modelo para obtener la distribución por clúster.
    """
    if pd is None:
        raise HTTPException(status_code=500, detail="Pandas no está disponible en este entorno.")
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado. No se puede analizar el archivo.")

    # leer archivo
    try:
        contents = file.file.read()
        file.file.seek(0)
        if file.filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents), parse_dates=["InvoiceDate"])
        else:
            # intentar excel
            df = pd.read_excel(io.BytesIO(contents), parse_dates=["InvoiceDate"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo leer el archivo: {e}")

    required = {"CustomerID", "InvoiceDate", "Quantity", "UnitPrice"}
    if not required.issubset(set(df.columns)):
        raise HTTPException(status_code=400, detail=f"Columnas requeridas faltantes. Se requieren: {sorted(required)}. Archivo tiene: {list(df.columns)}")

    # asegurar tipos
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    if df["InvoiceDate"].isna().any():
        raise HTTPException(status_code=400, detail="Algunas filas tienen fechas inválidas en 'InvoiceDate'. Usa formato YYYY-MM-DD o similar.")

    # Calcular por cliente dentro del periodo del archivo
    period_max = df["InvoiceDate"].max()

    # Total price por fila
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    agg = df.groupby("CustomerID").agg(
        last_purchase=("InvoiceDate", "max"),
        first_purchase=("InvoiceDate", "min"),
        Frequency=("InvoiceDate", "count"),
        Monetary=("TotalPrice", "sum"),
        TotalQuantity=("Quantity", "sum"),
    )
    agg["Recency"] = (period_max - agg["last_purchase"]).dt.days
    agg["CustomerLifespan"] = (agg["last_purchase"] - agg["first_purchase"]).dt.days
    # Evitar división por cero
    agg["AvgOrderValue"] = agg["Monetary"] / agg["Frequency"].replace(0, 1)

    # Preparar DataFrame de features en el orden esperado por el modelo
    cols = _feature_order(model) or ["Recency", "Frequency", "Monetary", "AvgOrderValue", "TotalQuantity", "CustomerLifespan"]
    features = agg.reset_index()[cols]

    # Asegurar no-null (llenar con 0s donde tenga sentido)
    features = features.fillna(0)

    # Predecir usando el modelo en lote
    try:
        preds = model.predict(features[cols])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir clusters para los clientes: {e}")

    pred_series = pd.Series(preds)
    counts = pred_series.value_counts(normalize=True).sort_index()

    # preparar salida en porcentaje (dos decimales)
    n_clusters = int(getattr(_detect_kmeans_component(model), "n_clusters", counts.index.max() + 1))
    distribution = {}
    for c in range(n_clusters):
        pct = float(counts.get(c, 0.0) * 100)
        distribution[str(c)] = round(pct, 2)

    total_customers = int(len(preds))

    # incluir perfiles si existen
    profile_snippet = None
    try:
        import json
        if PROFILES_PATH.exists():
            with open(PROFILES_PATH, "r", encoding="utf-8") as f:
                profiles = json.load(f)
            profile_snippet = {str(k): profiles.get(str(k)) for k in range(n_clusters) if str(k) in profiles}
    except Exception:
        profile_snippet = None

    return {
        "total_customers": total_customers,
        "distribution_percent": distribution,
        "period_end": str(period_max.date()),
        "profiles_available": bool(profile_snippet),
        "profiles": profile_snippet,
    }



@app.get("/upload", response_class=HTMLResponse)
def upload_page():
        """Página sencilla para subir CSV/XLSX y mostrar gráfico de distribución por clúster."""
        html = """
        <html>
        <head>
            <title>Analizar periodo - Subir CSV/XLSX</title>
            <style>
                body {font-family: Arial; max-width:820px; margin:auto; padding:18px}
                input[type=file] {display:block; margin-top:12px}
                button {margin-top:12px; background:#1976d2; color:#fff; border:0; padding:8px 12px; border-radius:6px}
                .info {margin-top:12px; color:#444}
                #chart-container {width:100%; max-width:600px; margin-top:18px}
            </style>
        </head>
        <body>
            <h2>Analizar periodo completo</h2>
            <p class='info'>Sube un archivo CSV o XLSX con transacciones (columnas mínimas: <code>CustomerID</code>, <code>InvoiceDate</code>, <code>Quantity</code>, <code>UnitPrice</code>).</p>
            <input id='file' type='file' accept='.csv, .xlsx, .xls' />
            <button id='uploadBtn' onclick='uploadFile()'>Subir y analizar</button>
            <div id='upload_result' class='info'></div>
            <div id='summary' class='info'></div>
            <div id='chart-container'>
                <canvas id='chart'></canvas>
            </div>

            <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
            <script>
            async function uploadFile(){
                const input = document.getElementById('file');
                if(!input.files || input.files.length === 0){ alert('Selecciona un archivo primero'); return; }
                const fd = new FormData();
                fd.append('file', input.files[0]);
                document.getElementById('upload_result').innerText = 'Subiendo y analizando...';
                try{
                        const res = await fetch('/analyze_upload', {method:'POST', body: fd});
                        if(!res.ok){ const txt = await res.text(); document.getElementById('upload_result').innerText = 'Error: '+txt; return; }
                        const data = await res.json();
                        document.getElementById('upload_result').innerText = '';
                        document.getElementById('summary').innerText = `Total clientes: ${data.total_customers} · Period end: ${data.period_end}`;
                        const dist = data.distribution_percent || {};
                        const profiles = data.profiles || {};

                        // Build labels using available profile info when present
                        const labels = Object.keys(dist).map(k=>{
                            const prof = profiles[k] || profiles[String(k)];
                            if(prof){
                                if(typeof prof === 'string') return `Cluster ${k}: ${prof}`;
                                if(prof.name) return `Cluster ${k}: ${prof.name}`;
                                if(prof.title) return `Cluster ${k}: ${prof.title}`;
                                if(prof.description) return `Cluster ${k}: ${prof.description}`;
                                // fallback to JSON snippet
                                return `Cluster ${k}: ${JSON.stringify(prof)}`;
                            }
                            // fallback to generic name
                            const fallback = {0: 'Clientes Estandar', 1: 'Inactivos de bajo valor', 2: 'Inactivos de bajo valor', 3: 'Clientes Valiosos'};
                            return `Cluster ${k}: ${fallback[k] || ('Cluster '+k)}`;
                        });

                        const values = Object.keys(dist).map(k=>dist[k]);
                        const colors = ['#4CAF50','#FFB74D','#90CAF9','#E57373','#BA68C8','#FFF176','#80CBC4'];
                        if(window.myChart) window.myChart.destroy();
                        const ctx = document.getElementById('chart').getContext('2d');
                        window.myChart = new Chart(ctx, {
                            type: 'pie',
                            data: { labels: labels, datasets:[{ data: values, backgroundColor: labels.map((_,i)=>colors[i % colors.length]) }] },
                            options: { responsive:true, plugins: { tooltip: { callbacks: { label: function(context){ return context.label + ' — ' + context.parsed + '%'; } } } } }
                        });

                        // show a brief profile legend below chart if profiles present
                        const legend = document.getElementById('upload_result');
                        if(Object.keys(profiles).length > 0){
                            let html = '<div style="margin-top:12px"><strong>Perfiles detectados:</strong><ul>';
                            for(const k of Object.keys(dist)){
                                const prof = profiles[k] || profiles[String(k)];
                                if(prof){
                                    let txt = '';
                                    if(typeof prof === 'string') txt = prof;
                                    else if(prof.description) txt = prof.description;
                                    else if(prof.name) txt = prof.name;
                                    else txt = JSON.stringify(prof);
                                    html += `<li><strong>Cluster ${k}:</strong> ${txt}</li>`;
                                }
                            }
                            html += '</ul></div>';
                            legend.innerHTML = html;
                        } else {
                            legend.innerText = '';
                        }
                    }catch(err){ document.getElementById('upload_result').innerText = 'Error: '+err; }
            }
            </script>
        </body>
        </html>
        """
        return HTMLResponse(html)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8002)
