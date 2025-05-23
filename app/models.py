import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from aequitas.group import Group
from sklearn.model_selection import train_test_split
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from aequitas.bias import Bias
from aequitas.fairness import Fairness


# ─── Filepaths ────────────────────────────────────────────────────────────────

# ─── Determine project root ───────────────────────────────────────────────────
#ROOT = Path(__file__).resolve().parent.parent
#MODEL_NO_PROXY_PATH     = "model_no_proxy.pkl"
#MODEL_WITH_PROXY_PATH   = "model_with_proxy.pkl"
#TRAIN_NO_PROXY_CSV      = "train_no_proxy.csv"
#TEST_NO_PROXY_CSV       = "test_no_proxy.csv"
TRAIN_BASE_CSV    = "app/train_with_proxy.csv"
TEST_BASE_CSV     = "app/test_with_proxy.csv"



# ─── Categorical Mappings (copied from your script) ──────────────────────────
MAPPINGS = {
    'Marital Status': {
        '1': 'single', '2': 'married', '3': 'widower', '4': 'divorced',
        '5': 'facto union', '6': 'legally separated'
    },
    'Application mode': {
        '1': '1st phase - general contingent', '2': 'Ordinance No. 612/93',
        '5': '1st phase - special contingent (Azores Island)',
        '7': 'Holders of other higher courses', '10': 'Ordinance No. 854-B/99',
        '15': 'International student (bachelor)', '16': '1st phase - special contingent (Madeira Island)',
        '17': '2nd phase - general contingent', '18': '3rd phase - general contingent',
        '26': 'Ordinance No. 533-A/99, item b2 (Different Plan)',
        '27': 'Ordinance No. 533-A/99, item b3 (Other Institution)',
        '39': 'Over 23 years old', '42': 'Transfer', '43': 'Change of course',
        '44': 'Technological specialization diploma holders',
        '51': 'Change of institution/course', '53': 'Short cycle diploma holders',
        '57': 'Change of institution/course (International)'
    },
    'Course': {
        '33': 'Biofuel Production Technologies', '171': 'Animation and Multimedia Design',
        '8014': 'Social Service (evening attendance)', '9003': 'Agronomy',
        '9070': 'Communication Design', '9085': 'Veterinary Nursing',
        '9119': 'Informatics Engineering', '9130': 'Equinculture', '9147': 'Management',
        '9238': 'Social Service', '9254': 'Tourism', '9500': 'Nursing',
        '9556': 'Oral Hygiene', '9670': 'Advertising and Marketing Management',
        '9773': 'Journalism and Communication', '9853': 'Basic Education',
        '9991': 'Management (evening attendance)'
    },
    "Daytime/evening attendance": {'1': 'daytime', '0': 'evening'},
    "Previous qualification": {
        '1': 'Secondary education', '2': 'Higher education - bachelor',
        '3': 'Higher education - degree', '4': 'Higher education - master',
        '5': 'Higher education - doctorate', '6': 'Frequency of higher education',
        '9': '12th year - not completed', '10': '11th year - not completed',
        '12': 'Other - 11th year of schooling', '14': '10th year of schooling',
        '15': '10th year - not completed', '19': 'Basic education 3rd cycle',
        '38': 'Basic education 2nd cycle', '39': 'Technological specialization course',
        '40': 'Higher education - degree (1st cycle)', '42': 'Professional higher technical course',
        '43': 'Higher education - master (2nd cycle)'
    },
    "Mother's qualification": {
        '1': 'Secondary Education', '2': 'Bachelor', '3': 'Degree',
        '4': 'Master', '5': 'Doctorate', '6': 'Frequency of Higher Education',
        '9': '12th Year - Not Completed', '10': '11th Year - Not Completed',
        '11': '7th Year (Old)', '12': 'Other - 11th Year', '14': '10th Year',
        '18': 'General commerce', '19': 'Basic Education 3rd Cycle',
        '22': 'Technical-professional', '26': '7th year',
        '27': '2nd cycle general high school', '29': '9th Year - Not Completed',
        '30': '8th year', '34': 'Unknown', '35': "Can't read/write",
        '36': 'Read without 4th year', '37': 'Basic education 1st cycle',
        '38': 'Basic education 2nd cycle', '39': 'Technological specialization',
        '40': 'Higher education - degree', '41': 'Specialized higher studies',
        '42': 'Professional higher technical', '43': 'Master', '44': 'Doctorate'
    },
    "Nationality": {
        '1': 'Portuguese', '2': 'German', '6': 'Spanish', '11': 'Italian',
        '13': 'Dutch', '14': 'English', '17': 'Lithuanian', '21': 'Angolan',
        '22': 'Cape Verdean', '24': 'Guinean', '25': 'Mozambican', '26': 'Santomean',
        '32': 'Turkish', '41': 'Brazilian', '62': 'Romanian', '100': 'Moldova',
        '101': 'Mexican', '103': 'Ukrainian', '105': 'Russian',
        '108': 'Cuban', '109': 'Colombian'
    },
    'Educational special needs': {'0': 'No', '1': 'Yes'},
    'Gender': {'0': 'Female', '1': 'Male'},
    'Age Code': {'1': 'under40', '0': '40plus'},
    'Debtor': {'0': 'No', '1': 'Yes'},
    'Displaced': {'0': 'No', '1': 'Yes'},
    'Tuition fees up to date': {'0': 'No', '1': 'Yes'},
    'Scholarship holder': {'0': 'No', '1': 'Yes'},
    'International': {'0': 'No', '1': 'Yes'},
    'label_value': {'0': 'Graduate', '1': 'Dropout'}
}


TEST_WITH_PROXY  = pd.read_csv(TEST_BASE_CSV)

FEATURES_WITH_PROXY = [c for c in TEST_WITH_PROXY.columns if c != "label_value"]
#PROXY_COLS          = [c for c in FEATURES_WITH_PROXY if c not in FEATURES_NO_PROXY]
SENSITIVE_ATTRS    = ["Age Code", "Educational special needs", "Gender", "Debtor"]

# ─── Helpers ──────────────────────────────────────────────────────────────────
def get_model_features():
    """Features used for prediction (no-proxy)."""
    return list(FEATURES_WITH_PROXY)

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baseline train/test split with *only* TO_DROP removed."""
    train = pd.read_csv(TRAIN_BASE_CSV)
    test  = pd.read_csv(TEST_BASE_CSV)
    return train, test


def numeric_feature_list(df: pd.DataFrame) -> list[str]:
    """All numeric, label excluded."""
    return df.select_dtypes(include=[np.number]).columns\
             .difference(["label_value"]).tolist()

def variant_features(
    base_feats: list[str],
    exclude_cols: list[str] | None = None
) -> list[str]:
    """
    Return the feature list *after* dropping `exclude_cols`.

    Parameters
    ----------
    base_feats   : every numeric feature in the baseline split
    exclude_cols : columns the user wants removed (grade, SES, or anything)

    If exclude_cols is None → nothing is dropped (“all features” variant).
    """
    if not exclude_cols:
        return base_feats
    return [c for c in base_feats if c not in set(exclude_cols)]


def train_and_eval_variant(
    exclude_cols: list[str] | None = None,
    threshold: float = .5,
) -> dict[str, Any]:
    """
    • Trains RF on baseline split with `exclude_cols` removed.
    • Returns accuracy, F1, and full Aequitas cross-tabs.

    Parameters
    ----------
    exclude_cols : list of columns to drop from the feature set, or None (keep all)
    threshold    : probability cutoff for converting scores to labels
    """
    train, test = load_splits()
    all_feats   = numeric_feature_list(train)
    feats       = variant_features(all_feats, exclude_cols)

    #rf = RandomForestClassifier(n_estimators=300,
                                #random_state=42,
                                #n_jobs=-1)
    rf = LogisticRegression()
    rf.fit(train[feats], train["label_value"])

    scores  = rf.predict_proba(test[feats])[:, 1]
    y_true  = test["label_value"].astype(int)
    y_pred  = (scores >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)

    # Aequitas – ensure sensitive cols are categorical strings
    df_aeq = test.copy()
    df_aeq["score"] = y_pred
    
    for feature in SENSITIVE_ATTRS:
        df_aeq[feature] = df_aeq[feature].astype(str).map(MAPPINGS[feature]).fillna(df_aeq[feature].astype(str))

    fairness = audit_with_aequitas(
        df_aeq[SENSITIVE_ATTRS + ["score", "label_value"]],
        SENSITIVE_ATTRS,
        threshold,
    )

    return {
        "dropped_cols"        : exclude_cols or [],
        "num_features"        : len(feats),
        "accuracy"            : acc,
        "f1_score"            : f1,
        "fairness_at_threshold": fairness,
    }

def get_representation_counts(df: pd.DataFrame, feature: str, group_col: str) -> Dict:
    """
    Return group-wise distribution data for one feature and one group.
    If feature == group_col, returns a simple count of each category with a uniform schema.
    """
    # Validate input columns
    if feature not in df.columns or group_col not in df.columns:
        raise ValueError(f"Feature '{feature}' or group '{group_col}' not found in dataset.")

    # Work on a copy to avoid side‐effects
    df = df.copy()

    # Apply mapping for the feature, if provided
    feat_map = MAPPINGS.get(feature)
    if feat_map:
        df[feature] = (
            df[feature]
            .astype(str)
            .map(feat_map)
            .fillna(df[feature].astype(str))
        )

    # Apply mapping for the group column, if provided
    grp_map = MAPPINGS.get(group_col)
    if grp_map:
        df[group_col] = (
            df[group_col]
            .astype(str)
            .map(grp_map)
            .fillna(df[group_col].astype(str))
        )

    # Special‐case: feature and group_col are the same
    if feature == group_col:
        vc = (
            df[feature]
            .value_counts()
            .rename_axis(feature)         # name the index so reset_index picks it up
            .reset_index(name="count")    # turn it into two columns: [feature, "count"]
        )
        # duplicate the feature column under group_col so schema stays uniform
        vc[group_col] = vc[feature]
        return {
            "type": "categorical",
            "counts": vc.to_dict(orient="records")
        }

    # Categorical branch: either non‐numeric feature or mapping applied
    if not pd.api.types.is_numeric_dtype(df[feature]) or feat_map:
        grouped = (
            df
            .groupby([feature, group_col])
            .size()
            .reset_index(name="count")
        )
        return {
            "type": "categorical",
            "counts": grouped.to_dict(orient="records")
        }

    # Numerical branch: return raw (feature, group) pairs
    records = df[[feature, group_col]].dropna().to_dict(orient="records")
    return {
        "type": "numerical",
        "values": records
    }


def audit_with_aequitas(
    df: pd.DataFrame,
    sensitive_attrs: List[str],
    threshold: float
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run Aequitas cross-tabs at the given score threshold,
    returning every metric for each group.

    Params:
    - df: must contain columns [*sensitive_attrs, 'score', 'label_value']
    - sensitive_attrs: list of categorical columns to audit
    - threshold: probability cutoff to binarize scores

    Returns:
    {
      'race_cat': [
         { 'attribute_value':'black', 'tpr':…, 'fpr':…, … },
         …
      ],
      'gender': [
         { 'attribute_value':'male',  'tpr':…, 'fpr':…, … },
         …
      ]
    }
    """
    g = Group()
    # plug in the user’s threshold here
    xtab, _ = g.get_crosstabs(df)

    b = Bias()
    bdf = b.get_disparity_predefined_groups(xtab, original_df=df,
                                        ref_groups_dict = {
                                                    'Gender': 'Female',            # Reference for gender
                                                    'Debtor': 'No',            # Reference for race
                                                    'Age Code': 'under40',
                                                    'Educational special needs': 'No'  # Reference for age category
                                                },
                                        alpha=0.05, check_significance=True,
                                        mask_significance=True)
    f = Fairness()
    fdf = f.get_group_value_fairness(bdf)
    #fdf[['attribute_name', 'attribute_value'] + absolute_metrics + b.list_disparities(fdf) + parity_detrminations]
    res: Dict[str, List[Dict[str, Any]]] = {}
    for attr in sensitive_attrs:
        rec = fdf[fdf["attribute_name"] == attr]
        # dump every column Aequitas has computed
        res[attr] = rec.to_dict(orient="records")
    return res


def resample_train(
        df_tr: pd.DataFrame,
        method: str,
        feature: str,
        label_col: str = "label_value",
        random_state: int = 42,
) -> pd.DataFrame:
    """
    Resample `df_tr` to mitigate representation bias.

    Parameters
    ----------
    df_tr : pd.DataFrame
        Training set.
    method : str
        One of {"baseline", "uniform_groups_over", "uniform_groups_down",
                "uniform_labels", "uniform_both"}.
    feature : str
        Protected-attribute column to balance on.
    label_col : str, default "label_value"
        Target variable column.
    """
    if method == "baseline":
        return df_tr.copy()

    rng = np.random.RandomState(random_state)
    groups = df_tr[feature].unique()
    labels = df_tr[label_col].unique()
    frames = []

    # --- 1 & 2.  Uniform group size --------------------------------------
    if method in ("uniform_groups_oversampled", "uniform_groups_downsampled"):
        counts = df_tr[feature].value_counts()
        target_n = counts.max() if method.endswith("oversampled") else counts.min()

        for g in groups:
            grp = df_tr[df_tr[feature] == g]
            size_gap = target_n - len(grp)
            if size_gap > 0:                               # need more rows
                extra = grp.sample(size_gap,
                                   replace=True,
                                   random_state=random_state)
                grp = pd.concat([grp, extra])
            elif size_gap < 0:                             # need fewer rows
                grp = grp.sample(target_n, random_state=random_state)
            frames.append(grp)

        return pd.concat(frames, ignore_index=True)

    # --- 3. Uniform label distribution *within* each group ---------------
    if method == "uniform_labels":
        for g in groups:
            grp = df_tr[df_tr[feature] == g]
            label_counts = grp[label_col].value_counts()
            target = label_counts.max()
            for l in labels:
                cell = grp[grp[label_col] == l]
                need = target - len(cell)
                if need > 0:
                    cell = pd.concat([
                        cell,
                        cell.sample(need, replace=True,
                                    random_state=random_state)
                    ])
                frames.append(cell)

        return pd.concat(frames, ignore_index=True)

    # --- 4. Uniform counts for every (group, label) pair -----------------
    if method == "uniform_both":
        cell_sizes = (df_tr.groupby([feature, label_col])
                            .size()
                            .to_dict())
        target = max(cell_sizes.values())

        for g in groups:
            for l in labels:
                cell = df_tr[(df_tr[feature] == g) &
                             (df_tr[label_col] == l)]
                need = target - len(cell)
                if need > 0:
                    cell = pd.concat([
                        cell,
                        cell.sample(need, replace=True,
                                    random_state=random_state)
                    ])
                frames.append(cell)

        return pd.concat(frames, ignore_index=True)

    raise ValueError(f"Unknown method: {method}")


def retrain_and_evaluate(
    rows: List[Dict[str, Any]],
    method: str,
    feature: str,
    exclude_cols: list[str] | None = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    1. load baseline train split
    2. drop `exclude_cols` (if any) → base features
    3. resample train on `feature` by `method`
    4. fit RF (baseline = un-resampled)
    5. compute representation counts for `feature`
    6. evaluate on `rows` at `threshold`
    7. run Aequitas audit

    Returns accuracy, F1, rep-counts, fairness metrics.
    """
    # 1) baseline split & feature list
    train_base, _ = load_splits()
    all_feats  = numeric_feature_list(train_base)
    feats      = variant_features(all_feats, exclude_cols)

    # 2) resample
    df_res = resample_train(train_base, method, feature)

    # 3) train
    #rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    
    rf = LogisticRegression()
    rf.fit(df_res[feats], df_res["label_value"])

    # 4) representation counts on *resampled* train

    rep_counts = get_representation_counts(df_res, feature, 'label_value')

    # 5) evaluation on supplied rows
    df_eval = pd.DataFrame(rows)
    scores  = rf.predict_proba(df_eval[feats])[:, 1]
    y_true  = df_eval["label_value"].astype(int)
    y_pred  = (scores >= threshold).astype(int)
    acc     = accuracy_score(y_true, y_pred)
    f1      = f1_score(y_true, y_pred)

    # 6) Aequitas audit
    df_aeq = df_eval.copy()
    df_aeq["score"] = y_pred

    for feature in SENSITIVE_ATTRS:
        df_aeq[feature] = df_aeq[feature].astype(str).map(MAPPINGS[feature]).fillna(df_aeq[feature].astype(str))

    fairness = audit_with_aequitas(
        df_aeq[SENSITIVE_ATTRS + ["score", "label_value"]],
        SENSITIVE_ATTRS,
        threshold,
    )

    return {
        "accuracy"             : acc,
        "f1_score"             : f1,
        "representation_counts": rep_counts,
        "fairness_at_threshold": fairness,
    }


def retrain_measurement_model(action: str, weight: float = 1.0):
    """
    action ∈ {"remove_proxy","include_proxy","downweight_proxy"}.
    - remove_proxy: use saved MODEL_NO_PROXY + TEST_NO_PROXY
    - include_proxy: use saved MODEL_WITH_PROXY + TEST_WITH_PROXY
    - downweight_proxy: re‐train RF on TRAIN_WITH_PROXY with proxies *= weight
    """

    if action == "downweight_proxy":
        # modify training set
        df_tr = TRAIN_WITH_PROXY.copy()
        for c in PROXY_COLS:
            df_tr[c] *= weight
        rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        rf.fit(df_tr[FEATURES_WITH_PROXY], df_tr["label_value"])
        df = TEST_WITH_PROXY.copy()
        feats = FEATURES_WITH_PROXY
    else:
        raise ValueError(f"Unknown action {action}")

    y = df["label_value"]
    yhat = rf.predict(df[feats])
    return {"accuracy": accuracy_score(y,yhat), "tpr": recall_score(y,yhat)}

def compute_group_list_sizes(
    feature: str,
    k: int = 1000,
    method: Optional[str] = None,
    exclude_cols: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    Compute top-k list sizes per subgroup of `feature`, optionally using a resampled train set.

    - If `method` is provided, resample the train split by that method.
    - Otherwise, use the original training data.
    """
    # 1) Load splits
    train_base, test_base = load_splits()
    all_feats = numeric_feature_list(train_base)
    feats     = variant_features(all_feats, exclude_cols)

    # 2) Optionally resample
    df_res = resample_train(train_base, method, feature) if method else train_base.copy()

    # 3) Train model with higher max_iter
    model = LogisticRegression(max_iter=1000)
    model.fit(df_res[feats], df_res["label_value"])

    # 4) Score held-out test split
    test_df = test_base.copy()
    test_df["score"] = model.predict_proba(test_df[feats])[:, 1]

    # 5) Compute rolling recall per subgroup
    groups = []
    for val, grp in test_df.groupby(feature):
        grp_sorted           = grp.sort_values("score", ascending=False).copy()
        total_pos            = grp_sorted["label_value"].sum()
        grp_sorted["cum_tp"] = grp_sorted["label_value"].cumsum()
        grp_sorted["cum_recall"] = (
            grp_sorted["cum_tp"] / total_pos if total_pos > 0 else 0
        )
        groups.append(grp_sorted)

    merged = pd.concat(groups, ignore_index=True)

    # 6) Select top-k by rolling recall, casting keys to strings
    topk  = merged.sort_values("cum_recall", ascending=False).head(k)
    sizes = {str(val): len(subgrp) for val, subgrp in topk.groupby(feature)}

    return sizes


def posthoc_group_threshold(
    feature: str,
    k: int = 1000,
    method: Optional[str] = None,
    exclude_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Apply post-hoc subgroup thresholds using computed top-k sizes:
    1) Compute sizes via compute_group_list_sizes
    2) Train model on (resampled) train set
    3) Score `rows`, apply subgroup sizes to predict
    4) Compute metrics and fairness audit
    """
    # 1) Compute group-specific list sizes
    sizes = compute_group_list_sizes(feature, k, method, exclude_cols)

    # 2) Train (resampled or original)
    train_base, test_base = load_splits()
    all_feats = numeric_feature_list(train_base)
    feats = variant_features(all_feats, exclude_cols)
    df_train = resample_train(train_base, method, feature) if method else train_base.copy()
    model = LogisticRegression().fit(df_train[feats], df_train["label_value"])

    # 3) Score and apply sizes
    df_eval = pd.DataFrame(test_base).copy()
    df_eval["score"] = model.predict_proba(df_eval[feats])[:, 1]
    df_eval["y_pred"] = 0
    for val, grp in df_eval.groupby(feature):
        n_select = sizes.get(val, 0)
        top_idx = grp.nlargest(n_select, "score").index
        df_eval.loc[top_idx, "y_pred"] = 1

    # 4) Metrics & audit
    y_true = df_eval["label_value"].astype(int)
    y_pred = df_eval["y_pred"].astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    rep_counts = get_representation_counts(df_train, feature, "label_value")

    df_aeq = df_eval.rename(columns={"y_pred": "score"})
    for attr in SENSITIVE_ATTRS:
        df_aeq[attr] = (
            df_aeq[attr]
            .astype(str)
            .map(MAPPINGS.get(attr, {}))
            .fillna(df_aeq[attr].astype(str))
        )
    fairness = audit_with_aequitas(
        df_aeq[SENSITIVE_ATTRS + ["score", "label_value"]],
        SENSITIVE_ATTRS,
        k
    )

    return {
        "group_list_sizes": sizes,
        "accuracy": acc,
        "f1_score": f1,
        "representation_counts": rep_counts,
        "fairness_at_threshold": fairness,
    }


def predict_risk(data: dict):
    """
    Single‐row prediction using the no‐proxy model.
    """
    df = pd.DataFrame([data])
    score = MODEL_NO_PROXY.predict_proba(df[FEATURES_NO_PROXY])[:,1][0]
    label = int(score >= 0.5)
    return score, label

