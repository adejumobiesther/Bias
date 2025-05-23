from fastapi import FastAPI, Request, Form, Body
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from typing import List, Dict, Any, Optional
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from models import (
    get_model_features,
    get_representation_counts,
    predict_risk,
    retrain_and_evaluate,
    audit_with_aequitas,
    load_splits,
    numeric_feature_list,
    variant_features,
    train_and_eval_variant,
    retrain_and_evaluate,
    compute_group_list_sizes,
    posthoc_group_threshold
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # or restrict to your html’s domain
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache feature list
FEATURES = get_model_features()

VALID_METHODS  = {"uniform_labels", "uniform_groups_oversampled", "uniform_groups_downsampled", "uniform_both"}
SENSITIVE_ATTRS    = ["Age Code", "Educational special needs", "Gender", "Debtor"]

gender_map = {1: 'male', 0: 'female'}
debtor_map = {1: "yes", 0: "no"}
ed_needs_map = {1: 'yes', 0: 'no'}
age_map = {1: "under40", 0: "40plus"}

train_base, test_base = load_splits()
BASE_FEATURES = numeric_feature_list(train_base)
BASE_MODEL    = RandomForestClassifier(
    n_estimators=300, random_state=42, n_jobs=-1
).fit(train_base[BASE_FEATURES], train_base["label_value"])

@app.get("/samples")
async def samples():
    """
    Returns all of your test-set rows as JSON so the table can populate.
    """
    return JSONResponse(content=FEATURES)

@app.post("/bias/label_distribution")
async def label_distribution(
    split: str = Body("train",  embed=True, description="Which split to use: 'train' or 'test'")
) -> Dict[str, Dict[int, int]]:
    """
    Returns the raw counts of label_value in the specified split.
    """
    train_base, test_base = load_splits()
    if split == 'train':
        counts = train_base["label_value"].value_counts().to_dict()
    else:
        counts = test_base["label_value"].value_counts().to_dict() # e.g. {0: 1234, 1: 567}
    
    return {"counts": counts}


@app.post("/predict")
async def predict(request: Request):
    """
    Read form fields, run predict_risk, return score & label.
    """
    form = await request.form()
    data = {k: float(v) for k, v in form.items() if k in FEATURES}
    score, label = predict_risk(data)
    return {"score": score, "label": label}

@app.post("/bias/representation")
async def rep_counts_dynamic(
    feature: str = Body(..., description="Feature to analyze"),
    group: str = Body(..., description="Group to compare by (e.g. Gender)"),
    split: str = Body(..., description="Split type to choose: 'train' or 'test'")
):
    """
    Returns raw distribution of a feature across group values.
    - If categorical, returns counts: {type: "categorical", counts: [...]}
    - If numerical, returns values: {type: "numerical", values: [...]}
    """
    df_train, df_test = load_splits()
    try:
        df = df_train if split == "train" else df_test
        result = get_representation_counts(df, feature, group)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/bias/representation/retrain")
async def representation_retrain(
    method : str                  = Body(...),
    feature: str                  = Body(...),
    exclude_cols: List[str] | None = Body(None,
        description="Columns to remove before training (grade / SES / custom)"),
    threshold: float = Body(0.5),
):
    if method not in VALID_METHODS:
        raise HTTPException(400, f"method must be one of {', '.join(VALID_METHODS)}")
    if feature not in SENSITIVE_ATTRS:
        raise HTTPException(400, f"feature must be one of {', '.join(SENSITIVE_ATTRS)}")
    
    df_train, df_test = load_splits()
    try:
        result = retrain_and_evaluate(
            rows        = df_test,
            method      = method,
            feature     = feature,
            exclude_cols= exclude_cols,
            threshold   = threshold,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Retraining failed: {e}")

    return JSONResponse(content=result)

@app.post("/predict-batch")
async def predict_batch(
    rows: List[Dict[str, Any]] = Body(...),
    threshold: float = Body(...)
):
    # 1) Build DataFrame from posted rows
    df      = pd.DataFrame(rows)
    X       = df[BASE_FEATURES]
    y_true  = df["label_value"].astype(int)
    scores  = BASE_MODEL.predict_proba(X)[:, 1]
    y_pred  = (scores >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)

    # make sensitive attributes categorical strings
    df_aeq = df.copy()
    df_aeq["score"] = scores
    for col, mapper in [
        ("Age Code", age_map),
        ("Debtor", debtor_map),
        ("Educational special needs", ed_needs_map),
        ("Gender", gender_map),
    ]:
        if col in df_aeq.columns:
            df_aeq[col] = df_aeq[col].map(mapper).fillna("other")

    fairness = audit_with_aequitas(
        df_aeq[SENSITIVE_ATTRS + ["score", "label_value"]],
        SENSITIVE_ATTRS,
        threshold,
    )

    return JSONResponse(content={
        "accuracy": acc,
        "f1_score": f1,
        "fairness_at_threshold": fairness,
    })

@app.post("/bias/measurement")
async def measurement_bias(
    exclude_cols: List[str] | None = Body(None,
    description="Which grade / SES (or other) columns to drop"),
    threshold: float = Body(0.5),
):
    """
    Trains a fresh RF after removing `exclude_cols` and returns performance +
    fairness metrics (no resampling).
    """
    result = train_and_eval_variant(
        exclude_cols = exclude_cols,
        threshold    = threshold,
    )
    return JSONResponse(content=result)


@app.post("/bias/posthoc/topk_sizes")
async def get_topk_sizes(
    feature: str = Body(...),
    k: int = Body(1000),
    method: Optional[str] = Body(None),
    exclude_cols: Optional[List[str]] = Body(None),
) -> Dict[str, int]:
    return compute_group_list_sizes(feature, k, method, exclude_cols)


@app.post("/bias/posthoc/group_threshold")
async def posthoc_group_threshold_endpoint(
    feature: str = Body(...),
    k: int = Body(1000),
    method: Optional[str] = Body(None),
    exclude_cols: Optional[List[str]] = Body(None),
) -> Dict[str, Any]:
    return posthoc_group_threshold(feature, k, method, exclude_cols)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
