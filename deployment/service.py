from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pickle

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

# Dependencias opcionales
try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

# Ruta esperada del modelo exportado desde el notebook
MODEL_PATH = Path(__file__).with_name("kmeans.pkl")
PROFILES_PATH = Path(__file__).parent.parent / "notebook" / "cluster_profiles.json"


def load_model(path: Path) -> Any:
    if not path.exists():
        return None
    # Intentar con joblib primero
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass
    # Fallback a pickle
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except pickle.UnpicklingError:
        with open(path, "rb") as f:
            return pickle.load(f, fix_imports=True, encoding="latin1")


model = load_model(MODEL_PATH)

app = FastAPI(title="Customer Segmentation KMeans API", version="1.0")


# Utilidades

def _detect_kmeans_component(obj: Any) -> Any:
    """Intenta localizar el componente KMeans dentro de un pipeline u objeto compuesto."""
    # sklearn Pipeline
    step = None
    named_steps = getattr(obj, "named_steps", None)
    if isinstance(named_steps, dict):
        # buscar por nombres comunes
        for key in ("kmeans", "model", "estimator", "cluster", "clf", "final"):
            if key in named_steps and hasattr(named_steps[key], "cluster_centers_"):
                step = named_steps[key]
                break
        # si no, buscar cualquiera con atributo de centros
        if step is None:
            for comp in named_steps.values():
                if hasattr(comp, "cluster_centers_"):
                    step = comp
                    break
    # si el propio objeto es un KMeans
    if step is None and hasattr(obj, "cluster_centers_"):
        step = obj
    return step


def _feature_order(obj: Any) -> Optional[List[str]]:
    # Intentar extraer nombres de features si existen
    for candidate in (obj, getattr(obj, "named_steps", None), _detect_kmeans_component(obj)):
        if candidate is None:
            continue
        cols = getattr(candidate, "feature_names_in_", None)
        if cols is not None:
            return list(cols)
    # A veces se guarda manualmente
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

    # Construir entrada en el orden esperado
    if pd is not None:
        X = pd.DataFrame([[values.get(c) for c in cols]], columns=cols)
    else:
        X = [[values.get(c) for c in cols]]

    # Predicción
    pred = None
    try:
        pred = model.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir con el modelo: {e}")

    label = int(pred[0]) if hasattr(pred, "__iter__") else int(pred)

    # Distancias a centroides si es posible
    km = _detect_kmeans_component(model)
    distances = None
    try:
        if km is not None and hasattr(km, "transform"):
            # Si es pipeline, usar pipeline.transform para considerar scaler
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


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "has_kmeans": bool(_detect_kmeans_component(model)) if model is not None else False,
    }


@app.get("/", response_class=HTMLResponse)
def form(result: Optional[str] = None):
    style = """
    body {font-family: Arial; max-width: 700px; margin: auto; padding: 16px}
    label {display: block; margin-top: 10px}
    input {width: 100%; padding: 8px; margin-top: 6px}
    button {margin-top: 16px; background: #4CAF50; color: #fff; border: 0; padding: 10px 16px; border-radius: 4px; cursor: pointer}
    .warn {background: #fff3cd; padding: 10px; border: 1px solid #ffeeba; border-radius: 4px; margin-bottom: 12px}
    """
    warn = ""
    if model is None:
        warn = f"<div class='warn'>Modelo no encontrado en {MODEL_PATH.name}. Exporta el pipeline desde el notebook como 'kmeans_rfm_pipeline.pkl'.</div>"

    return f"""
    <html>
    <head><title>KMeans Segmentation</title><style>{style}</style></head>
    <body>
      <h2>Segmentación de Clientes (KMeans)</h2>
      {warn}
      <form action="/" method="post">
        <label>Recencia</label>
        <input name="Recency" type="number" step="any" required />
        <label>Frecuencia</label>
        <input name="Frequency" type="number" step="any" required />
        <label>Valor total gastado</label>
        <input name="Monetary" type="number" step="any" required />
        <label>Promedio Gastado por Compra</label>
        <input name="AvgOrderValue" type="number" step="any" required />
        <label>Cantidad de unidades compradas</label>
        <input name="TotalQuantity" type="number" step="any" required />
        <label>Tiempo entre la primera y última compra</label>
        <input name="CustomerLifespan" type="number" step="any" required />
        <button type="submit">Predecir clúster</button>
      </form>
      <hr />
      {f"<pre>{result}</pre>" if result else ""}
    </body></html>
    """


@app.post("/", response_class=HTMLResponse)
def predict_form(
    Recency: float = Form(...),
    Frequency: float = Form(...),
    Monetary: float = Form(...),
    AvgOrderValue: float = Form(...),
    TotalQuantity: float = Form(...),
    CustomerLifespan: float = Form(...),
):
    global cluster_name
    values = {
        "Recency": Recency,
        "Frequency": Frequency,
        "Monetary": Monetary,
        "AvgOrderValue": AvgOrderValue,
        "TotalQuantity": TotalQuantity,
        "CustomerLifespan": CustomerLifespan,
    }
    out = _predict_cluster(values)

    match out["cluster"]:
        case 0:
            cluster_name = "Cluster 0: Clientes Estandar"
        case 1:
            cluster_name = "Cluster 1: Clientes Inactivos de bajo valor"
        case 2:
            cluster_name = "Cluster 2: Clientes Inactivos de bajo valor"
        case 3:
            cluster_name = "Cluster 3: Clientes Valiosos"


    details = [
        f"Cluster predicho: {out['cluster']} - {cluster_name}",
        f"Orden de features: {', '.join(out['feature_order'])}",
    ]
    if out.get("n_clusters"):
        details.append(f"Total clusters en el modelo: {out['n_clusters']}")
    if out.get("distances") is not None:
        d_str = ", ".join(f"{i}:{v:.4f}" for i, v in enumerate(out["distances"]))
        details.append(f"Distancias a centroides: [{d_str}]")

    return form(result="\n".join(details))


@app.post("/predict")
def predict_json(payload: Dict[str, float]):
    # Espera un JSON con las 6 claves
    out = _predict_cluster(payload)
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



if __name__ == "__main__":
    # Modo local (por ejemplo: python deployment/service.py)
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
