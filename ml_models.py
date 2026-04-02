"""
ml_models.py
Trains and evaluates the full TE Draft Intelligence model suite:
  1. Success Probability  — GBM + RF + Logistic ensemble
  2. Draft Pick Predictor — GBM regression
  3. Career EPA Regressor — GBM regression
  4. SHAP Explainability  — per-player feature attributions
  5. Clustering          — prospect archetypes (KMeans)
"""

import numpy as np
import pandas as pd
import json, pickle, os
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, mean_absolute_error
import shap
import warnings
warnings.filterwarnings("ignore")

from data_pipeline import build_dataset, FEATURES, FEATURE_LABELS

# ── paths ──────────────────────────────────────────────────────────────────
MODEL_DIR   = "models"
RESULTS_DIR = "results"
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── utilities ──────────────────────────────────────────────────────────────

def _impute(X: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    return X.fillna(medians)


def _get_medians(X: pd.DataFrame) -> pd.Series:
    return X.median()


# ═══════════════════════════════════════════════════════════════════════════
# 1. SUCCESS PROBABILITY MODEL (ensemble)
# ═══════════════════════════════════════════════════════════════════════════

def train_success_model(X_tr, y_tr):
    """Soft-voting ensemble: GBM + RF + Logistic. Also returns fitted bare GBM for SHAP."""
    gbm_bare = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.04,
        subsample=0.8, min_samples_leaf=5, random_state=42
    )
    gbm_bare.fit(X_tr, y_tr)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=0.5, max_iter=500, random_state=42))
    ])
    # Use a fresh (unfitted) GBM clone in the ensemble so VotingClassifier fits it internally
    gbm_clone = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.04,
        subsample=0.8, min_samples_leaf=5, random_state=42
    )
    ensemble = VotingClassifier(
        estimators=[("gbm", gbm_clone), ("rf", rf), ("lr", lr)],
        voting="soft",
        weights=[3, 2, 1],
    )
    ensemble.fit(X_tr, y_tr)
    return ensemble, gbm_bare   # bare GBM for SHAP


def evaluate_success_model(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    accs = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return {
        "cv_auc_mean":  round(float(aucs.mean()), 4),
        "cv_auc_std":   round(float(aucs.std()),  4),
        "cv_acc_mean":  round(float(accs.mean()), 4),
        "cv_acc_std":   round(float(accs.std()),  4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. DRAFT PICK PREDICTOR (regression)
# ═══════════════════════════════════════════════════════════════════════════

PICK_FEATURES = [
    "40yd", "Wt", "height_in", "Vertical", "Broad Jump", "3Cone", "Shuttle",
    "cfb_rec_yds", "cfb_rec_td", "cfb_rec", "cfb_ypr",
    "cfb_ppa_total", "cfb_usage_pass", "cfb_rec_yds_per_season",
]

def train_pick_model(df_labeled):
    has_pick = df_labeled[df_labeled["pick"].notna()].copy()
    X = has_pick[PICK_FEATURES].fillna(has_pick[PICK_FEATURES].median())
    y = has_pick["pick"]
    gbr = GradientBoostingRegressor(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    gbr.fit(X, y)
    medians = has_pick[PICK_FEATURES].median()
    mae = cross_val_score(gbr, X, y, cv=5, scoring="neg_mean_absolute_error")
    print(f"  Pick model CV MAE: {-mae.mean():.1f} ± {mae.std():.1f}")
    return gbr, medians


# ═══════════════════════════════════════════════════════════════════════════
# 3. CAREER EPA REGRESSOR
# ═══════════════════════════════════════════════════════════════════════════

EPA_FEATURES = [
    "40yd", "Wt", "height_in", "Vertical", "Broad Jump", "3Cone", "Shuttle",
    "pick", "cfb_rec_yds", "cfb_rec_td", "cfb_rec",
    "cfb_ypr", "cfb_ppa_total", "cfb_ppa_pass",
    "cfb_usage_pass", "cfb_rec_yds_per_season",
]

def train_epa_model(df_labeled):
    has_epa = df_labeled[df_labeled["career_epa"].notna()].copy()
    X = has_epa[EPA_FEATURES].fillna(has_epa[EPA_FEATURES].median())
    y = has_epa["career_epa"]
    gbr = GradientBoostingRegressor(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    gbr.fit(X, y)
    mae = cross_val_score(gbr, X, y, cv=5, scoring="neg_mean_absolute_error")
    print(f"  EPA model  CV MAE: {-mae.mean():.1f} ± {mae.std():.1f}")
    return gbr, has_epa[EPA_FEATURES].median()


# ═══════════════════════════════════════════════════════════════════════════
# 4. SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════

def compute_shap(gbm_bare, X_train_imp, X_all_imp, feature_names):
    """Returns SHAP values for every row in X_all_imp."""
    explainer = shap.TreeExplainer(gbm_bare)
    sv = explainer.shap_values(X_all_imp)
    return sv, explainer.expected_value


def shap_for_player(shap_values, player_idx, feature_names):
    """Return sorted list of (feature_label, shap_val) for one player."""
    sv = shap_values[player_idx]
    pairs = sorted(zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True)
    return [(FEATURE_LABELS.get(f, f), round(float(v), 4)) for f, v in pairs[:8]]


# ═══════════════════════════════════════════════════════════════════════════
# 5. PROSPECT ARCHETYPES (clustering)
# ═══════════════════════════════════════════════════════════════════════════

CLUSTER_FEATURES = [
    "40yd", "Wt", "height_in", "Vertical", "Broad Jump",
    "cfb_rec_yds_per_season", "cfb_ypr", "cfb_usage_pass",
]

ARCHETYPE_NAMES = {
    0: "Inline Blocker",
    1: "Receiving Weapon",
    2: "Move TE / H-Back",
    3: "Athletic Freak",
    4: "Balanced Starter",
}

def fit_clusters(X_imp, n=5):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_imp)
    km = KMeans(n_clusters=n, random_state=42, n_init=20)
    labels = km.fit_predict(Xs)
    return km, scaler, labels


def name_cluster(km, scaler, cluster_id, feature_names):
    """Return a human-readable description of a cluster centroid."""
    center = scaler.inverse_transform(km.cluster_centers_[cluster_id:cluster_id+1])[0]
    d = dict(zip(feature_names, center))
    return d


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — train everything, export results
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=== TE Draft Intelligence · ML Pipeline ===\n")

    # ── Load data ──────────────────────────────────────────────────────────
    print("[1/6] Loading & merging data...")
    train_df, df_2026 = build_dataset()
    labeled = train_df[train_df["success"].notna()].copy()
    print(f"      Labelled players : {len(labeled)}")
    print(f"      Success rate     : {labeled['success'].mean():.1%}")
    print(f"      2026 prospects   : {len(df_2026)}")

    # ── Medians (imputation) ───────────────────────────────────────────────
    X_all_raw  = train_df[FEATURES]
    medians    = _get_medians(X_all_raw)

    X_labeled  = labeled[FEATURES]
    X_labeled_imp = _impute(X_labeled, medians)
    y          = labeled["success"]

    X_all_imp  = _impute(X_all_raw, medians)
    X_2026_imp = _impute(df_2026[FEATURES], medians)

    # ── 1. Success model ───────────────────────────────────────────────────
    print("\n[2/6] Training success probability ensemble...")
    ensemble, gbm_bare = train_success_model(X_labeled_imp, y)
    metrics = evaluate_success_model(ensemble, X_labeled_imp, y)
    print(f"      CV AUC : {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")
    print(f"      CV Acc : {metrics['cv_acc_mean']:.3f} ± {metrics['cv_acc_std']:.3f}")

    # Predict all historical + 2026
    train_df["ml_success_prob"] = ensemble.predict_proba(X_all_imp)[:, 1]
    df_2026["ml_success_prob"]  = ensemble.predict_proba(X_2026_imp)[:, 1]

    # ── 2. Pick predictor ──────────────────────────────────────────────────
    print("\n[3/6] Training draft pick predictor...")
    pick_model, pick_medians = train_pick_model(train_df)
    X_2026_pick = _impute(df_2026[PICK_FEATURES], pick_medians)
    df_2026["ml_projected_pick"] = pick_model.predict(X_2026_pick).round(0).astype(int)

    X_all_pick = _impute(train_df[PICK_FEATURES], pick_medians)
    train_df["ml_projected_pick"] = pick_model.predict(X_all_pick).round(0).astype(int)

    # ── 3. EPA regressor ───────────────────────────────────────────────────
    print("\n[4/6] Training career EPA regressor...")
    epa_model, epa_medians = train_epa_model(train_df)
    X_2026_epa = _impute(df_2026[EPA_FEATURES], epa_medians)
    df_2026["ml_projected_epa"] = epa_model.predict(X_2026_epa).round(1)
    X_all_epa = _impute(train_df[EPA_FEATURES], epa_medians)
    train_df["ml_projected_epa"] = epa_model.predict(X_all_epa).round(1)

    # ── 4. SHAP ─────────────────────────────────────────────────────────────
    print("\n[5/6] Computing SHAP explanations...")
    shap_vals, base_val = compute_shap(gbm_bare, X_labeled_imp, X_all_imp, FEATURES)

    # Feature importance from SHAP (mean |shap|)
    fi_shap = np.abs(shap_vals).mean(axis=0)
    fi_list = sorted(
        [{"name": FEATURE_LABELS.get(f, f), "raw": f, "val": round(float(v), 5),
          "type": "college" if "cfb" in f else "combine"}
         for f, v in zip(FEATURES, fi_shap)],
        key=lambda x: -x["val"]
    )

    # Per-player SHAP top factors (for all historical players)
    player_shap = {}
    for i, row in train_df.iterrows():
        idx = train_df.index.get_loc(i)
        if idx < len(shap_vals):
            sv = shap_vals[idx]
            pairs = sorted(zip(FEATURES, sv), key=lambda x: abs(x[1]), reverse=True)[:6]
            player_shap[str(row["Player"])] = [
                {"feature": FEATURE_LABELS.get(f, f), "shap": round(float(v), 3)}
                for f, v in pairs
            ]

    # SHAP for 2026
    shap_vals_2026, _ = compute_shap(gbm_bare, X_labeled_imp, X_2026_imp, FEATURES)
    prospect_shap = {}
    for i, row in df_2026.iterrows():
        idx = df_2026.index.get_loc(i)
        if idx < len(shap_vals_2026):
            sv = shap_vals_2026[idx]
            pairs = sorted(zip(FEATURES, sv), key=lambda x: abs(x[1]), reverse=True)[:6]
            prospect_shap[str(row["Player"])] = [
                {"feature": FEATURE_LABELS.get(f, f), "shap": round(float(v), 3)}
                for f, v in pairs
            ]

    # ── 5. Clustering ──────────────────────────────────────────────────────
    print("\n[6/6] Fitting prospect archetypes (KMeans k=5)...")
    # Use all players with enough data
    cluster_mask = train_df[CLUSTER_FEATURES].notna().all(axis=1)
    X_clust_raw  = train_df.loc[cluster_mask, CLUSTER_FEATURES]
    X_clust_2026 = df_2026[CLUSTER_FEATURES].fillna(df_2026[CLUSTER_FEATURES].median())

    km, km_scaler, cluster_labels = fit_clusters(
        pd.concat([X_clust_raw, X_clust_2026], ignore_index=True)
    )

    # Assign cluster labels back
    all_cluster_labels = km.labels_
    n_hist = cluster_mask.sum()
    train_df.loc[cluster_mask, "archetype_id"] = all_cluster_labels[:n_hist]
    df_2026["archetype_id"] = all_cluster_labels[n_hist:]

    # Auto-name clusters by centroid characteristics
    archetype_names = _auto_name_clusters(km, km_scaler, CLUSTER_FEATURES)
    print(f"      Archetypes: {archetype_names}")

    # ── Export JSON ────────────────────────────────────────────────────────
    print("\n── Exporting results...")

    def safe(v):
        if v is None or (isinstance(v, float) and np.isnan(v)): return None
        if isinstance(v, (np.integer,)):  return int(v)
        if isinstance(v, (np.floating,)): return round(float(v), 4)
        return v

    # Historical players
    players_out = []
    for _, r in train_df.iterrows():
        nm = str(r["Player"])
        players_out.append({
            "name":             nm,
            "school":           str(r.get("College", "")),
            "pick":             safe(r.get("pick")),
            "round":            safe(r.get("round")),
            "draft_year":       safe(r.get("draft_year")),
            "forty":            safe(r.get("40yd")),
            "weight":           safe(r.get("Wt")),
            "height":           safe(r.get("height_in")),
            "vertical":         safe(r.get("Vertical")),
            "broad":            safe(r.get("Broad Jump")),
            "cone":             safe(r.get("3Cone")),
            "shuttle":          safe(r.get("Shuttle")),
            "cfb_rec_yds":      safe(r.get("cfb_rec_yds")),
            "cfb_rec_td":       safe(r.get("cfb_rec_td")),
            "cfb_rec":          safe(r.get("cfb_rec")),
            "cfb_ypr":          safe(r.get("cfb_ypr")),
            "cfb_ppa":          safe(r.get("cfb_ppa_total")),
            "cfb_usage_pass":   safe(r.get("cfb_usage_pass")),
            "career_rec_yds":   safe(r.get("career_rec_yds")),
            "career_rec_td":    safe(r.get("career_rec_td")),
            "career_epa":       safe(r.get("career_epa")),
            "career_games":     safe(r.get("career_games")),
            "career_fantasy":   safe(r.get("career_fantasy")),
            "success":          safe(r.get("success")),
            "ml_prob":          safe(r.get("ml_success_prob")),
            "ml_projected_pick":safe(r.get("ml_projected_pick")),
            "ml_projected_epa": safe(r.get("ml_projected_epa")),
            "archetype_id":     safe(r.get("archetype_id")),
            "archetype":        archetype_names.get(int(r["archetype_id"]) if pd.notna(r.get("archetype_id")) else -1, "Unknown"),
            "shap_factors":     player_shap.get(nm, []),
        })

    # 2026 prospects
    prospects_out = []
    for _, r in df_2026.sort_values("ml_success_prob", ascending=False).iterrows():
        nm = str(r["Player"])
        prospects_out.append({
            "name":              nm,
            "school":            str(r.get("College", r.get("School", ""))),
            "forty":             safe(r.get("40yd")),
            "weight":            safe(r.get("Wt")),
            "height":            safe(r.get("height_in")),
            "vertical":          safe(r.get("Vertical")),
            "broad":             safe(r.get("Broad Jump")),
            "cone":              safe(r.get("3Cone")),
            "shuttle":           safe(r.get("Shuttle")),
            "cfb_rec_yds":       safe(r.get("cfb_rec_yds")),
            "cfb_rec_td":        safe(r.get("cfb_rec_td")),
            "cfb_rec":           safe(r.get("cfb_rec")),
            "cfb_ypr":           safe(r.get("cfb_ypr")),
            "cfb_ppa":           safe(r.get("cfb_ppa_total")),
            "cfb_usage_pass":    safe(r.get("cfb_usage_pass")),
            "cfb_rec_yds_per_season": safe(r.get("cfb_rec_yds_per_season")),
            "ml_prob":           safe(r.get("ml_success_prob")),
            "ml_projected_pick": safe(r.get("ml_projected_pick")),
            "ml_projected_epa":  safe(r.get("ml_projected_epa")),
            "archetype_id":      safe(r.get("archetype_id")),
            "archetype":         archetype_names.get(int(r["archetype_id"]) if pd.notna(r.get("archetype_id")) else -1, "Unknown"),
            "shap_factors":      prospect_shap.get(nm, []),
        })

    # Save all result files
    results = {
        "model_metrics":     metrics,
        "feature_importance": fi_list,
        "archetype_names":   archetype_names,
        "base_shap_value":   round(float(np.array(base_val).flat[0]), 4),
    }
    with open(f"{RESULTS_DIR}/players.json",   "w") as f: json.dump(players_out, f)
    with open(f"{RESULTS_DIR}/prospects.json", "w") as f: json.dump(prospects_out, f)
    with open(f"{RESULTS_DIR}/model_info.json","w") as f: json.dump(results, f, indent=2)

    # Save pickled models for API server
    with open(f"{MODEL_DIR}/ensemble.pkl",   "wb") as f: pickle.dump(ensemble,   f)
    with open(f"{MODEL_DIR}/pick_model.pkl", "wb") as f: pickle.dump(pick_model, f)
    with open(f"{MODEL_DIR}/epa_model.pkl",  "wb") as f: pickle.dump(epa_model,  f)
    with open(f"{MODEL_DIR}/medians.pkl",    "wb") as f: pickle.dump({
        "success": medians, "pick": pick_medians, "epa": epa_medians
    }, f)
    with open(f"{MODEL_DIR}/km.pkl",         "wb") as f: pickle.dump((km, km_scaler, archetype_names), f)

    print(f"\n✓  {len(players_out)} historical players → results/players.json")
    print(f"✓  {len(prospects_out)} 2026 prospects   → results/prospects.json")
    print(f"✓  Model info                         → results/model_info.json")
    print(f"✓  Pickled models                     → models/")
    print(f"\nModel AUC: {metrics['cv_auc_mean']:.3f}")


def _auto_name_clusters(km, scaler, feature_names):
    """Heuristically name clusters from centroid values."""
    names = {}
    centers = scaler.inverse_transform(km.cluster_centers_)
    idx_40  = feature_names.index("40yd")         if "40yd" in feature_names else None
    idx_wt  = feature_names.index("Wt")           if "Wt"   in feature_names else None
    idx_rec = feature_names.index("cfb_rec_yds_per_season") if "cfb_rec_yds_per_season" in feature_names else None

    cluster_data = []
    for c, center in enumerate(centers):
        d = {f: center[i] for i, f in enumerate(feature_names)}
        cluster_data.append((c, d))

    # Sort clusters by key metrics to assign archetypes
    by_speed  = sorted(cluster_data, key=lambda x: x[1].get("40yd", 99))
    by_weight = sorted(cluster_data, key=lambda x: -x[1].get("Wt",   0))
    by_rec    = sorted(cluster_data, key=lambda x: -x[1].get("cfb_rec_yds_per_season", 0))

    labels = ["Receiving Weapon", "Athletic Freak", "Balanced Starter", "Inline Blocker", "Move TE / H-Back"]
    assigned = {}
    assigned[by_rec[0][0]]    = "Receiving Weapon"
    assigned[by_speed[0][0]]  = "Athletic Freak"
    assigned[by_weight[0][0]] = "Inline Blocker"
    for c, _ in cluster_data:
        if c not in assigned:
            if len([v for v in assigned.values() if v == "Move TE / H-Back"]) == 0:
                assigned[c] = "Move TE / H-Back"
            else:
                assigned[c] = "Balanced Starter"
    return {int(k): v for k, v in assigned.items()}


if __name__ == "__main__":
    main()
