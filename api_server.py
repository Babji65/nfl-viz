"""
api_server.py
Flask REST API that serves the ML models to the frontend.
Start with:  python api_server.py

Endpoints:
  GET  /api/status            — health check + model metrics
  GET  /api/players           — all historical player records
  GET  /api/prospects         — 2026 draft prospects
  GET  /api/model_info        — feature importance, metrics
  POST /api/predict           — custom player prediction
  GET  /api/similar/<name>    — find similar historical comparps
  GET  /api/player/<name>     — single player detail + SHAP
  GET  /api/archetypes        — cluster summary stats
"""

import os, json, pickle
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

RESULTS_DIR = "results"
MODEL_DIR   = "models"

# ── Load precomputed results ────────────────────────────────────────────────
def _load(fname):
    path = os.path.join(RESULTS_DIR, fname)
    with open(path) as f:
        return json.load(f)

try:
    PLAYERS    = _load("players.json")
    PROSPECTS  = _load("prospects.json")
    MODEL_INFO = _load("model_info.json")
    PLAYERS_IDX    = {p["name"]: p for p in PLAYERS}
    PROSPECTS_IDX  = {p["name"]: p for p in PROSPECTS}
    print(f"Loaded {len(PLAYERS)} players, {len(PROSPECTS)} prospects")
except FileNotFoundError:
    print("WARNING: Run ml_models.py first to generate results/")
    PLAYERS = PROSPECTS = MODEL_INFO = []
    PLAYERS_IDX = PROSPECTS_IDX = {}

# ── Load pickled models ─────────────────────────────────────────────────────
try:
    with open(f"{MODEL_DIR}/ensemble.pkl",   "rb") as f: ensemble   = pickle.load(f)
    with open(f"{MODEL_DIR}/pick_model.pkl", "rb") as f: pick_model = pickle.load(f)
    with open(f"{MODEL_DIR}/epa_model.pkl",  "wb" if False else "rb") as f: epa_model  = pickle.load(f)
    with open(f"{MODEL_DIR}/medians.pkl",    "rb") as f: MEDIANS    = pickle.load(f)
    with open(f"{MODEL_DIR}/km.pkl",         "rb") as f: km, km_scaler, archetype_names = pickle.load(f)
    MODELS_LOADED = True
    print("Models loaded")
except Exception as e:
    print(f"WARNING: Could not load pickled models ({e}). Run ml_models.py first.")
    MODELS_LOADED = False
    MEDIANS = {}

from data_pipeline import FEATURES, FEATURE_LABELS, PICK_FEATURES, EPA_FEATURES, CLUSTER_FEATURES


# ── helpers ─────────────────────────────────────────────────────────────────

def _safe(v):
    if v is None: return None
    try:
        if np.isnan(float(v)): return None
        return round(float(v), 4) if isinstance(v, float) else v
    except Exception:
        return v


def _impute_row(d: dict, medians: pd.Series) -> pd.DataFrame:
    row = pd.DataFrame([d])
    for col in medians.index:
        if col not in row.columns or pd.isna(row[col].iloc[0]):
            row[col] = medians[col]
    return row


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/api/status")
def status():
    return jsonify({
        "ok": True,
        "players":   len(PLAYERS),
        "prospects": len(PROSPECTS),
        "models_loaded": MODELS_LOADED,
        "model_auc": MODEL_INFO.get("model_metrics", {}).get("cv_auc_mean") if MODEL_INFO else None,
    })


@app.route("/api/players")
def get_players():
    # Optional filters via query params
    name   = request.args.get("name",   "").lower()
    year   = request.args.get("year",   "")
    school = request.args.get("school", "").lower()

    out = PLAYERS
    if name:
        out = [p for p in out if name in p["name"].lower()]
    if year:
        out = [p for p in out if str(p.get("draft_year","")) == year]
    if school:
        out = [p for p in out if school in str(p.get("school","")).lower()]
    return jsonify(out)


@app.route("/api/prospects")
def get_prospects():
    return jsonify(PROSPECTS)


@app.route("/api/model_info")
def get_model_info():
    return jsonify(MODEL_INFO)


@app.route("/api/player/<path:name>")
def get_player(name):
    p = PLAYERS_IDX.get(name) or PROSPECTS_IDX.get(name)
    if not p:
        return jsonify({"error": "Player not found"}), 404
    return jsonify(p)


# ── /api/similar/<name> — nearest neighbors by feature space ────────────────

@app.route("/api/similar/<path:name>")
def get_similar(name):
    target = PLAYERS_IDX.get(name) or PROSPECTS_IDX.get(name)
    if not target:
        return jsonify({"error": "Player not found"}), 404

    KEYS = ["forty", "weight", "height", "vertical", "broad",
            "cfb_rec_yds", "cfb_ypr", "cfb_usage_pass"]

    def vec(p):
        return np.array([float(p.get(k) or 0) for k in KEYS])

    tv = vec(target)

    scored = []
    for p in PLAYERS:
        if p["name"] == name:
            continue
        if p.get("career_rec_yds") is None:
            continue
        pv = vec(p)
        # Cosine similarity on non-zero entries
        norm = (np.linalg.norm(tv) * np.linalg.norm(pv))
        if norm == 0:
            continue
        sim = float(np.dot(tv, pv) / norm)
        scored.append({**p, "_sim": round(sim, 4)})

    scored.sort(key=lambda x: -x["_sim"])
    return jsonify(scored[:8])


# ── /api/predict — custom prediction for arbitrary player stats ─────────────

@app.route("/api/predict", methods=["POST"])
def predict():
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded. Run ml_models.py first."}), 503

    data = request.get_json(force=True)

    # Map incoming field names → internal names
    field_map = {
        "forty": "40yd", "weight": "Wt", "height": "height_in",
        "vertical": "Vertical", "broad": "Broad Jump",
        "cone": "3Cone", "shuttle": "Shuttle",
        "cfb_rec_yds": "cfb_rec_yds", "cfb_rec_td": "cfb_rec_td",
        "cfb_rec": "cfb_rec", "cfb_ypr": "cfb_ypr",
        "cfb_ppa": "cfb_ppa_total", "cfb_usage_pass": "cfb_usage_pass",
        "pick": "pick",
    }
    row = {}
    for ext, internal in field_map.items():
        v = data.get(ext)
        row[internal] = float(v) if v is not None else None

    # Derived
    seasons = float(data.get("cfb_seasons", 3) or 3)
    row["cfb_rec_yds_per_season"] = (row.get("cfb_rec_yds") or 0) / seasons
    row["cfb_rec_td_per_season"]  = (row.get("cfb_rec_td") or 0) / seasons
    row["cfb_rush_yds"]           = float(data.get("cfb_rush_yds", 0) or 0)
    row["cfb_ppa_pass"]           = row.get("cfb_ppa_total") or 0

    X_succ = _impute_row({f: row.get(f) for f in FEATURES},            MEDIANS["success"])
    X_pick = _impute_row({f: row.get(f) for f in PICK_FEATURES},       MEDIANS["pick"])
    X_epa  = _impute_row({f: row.get(f) for f in EPA_FEATURES},        MEDIANS["epa"])

    prob      = float(ensemble.predict_proba(X_succ[FEATURES])[0, 1])
    proj_pick = int(pick_model.predict(X_pick[PICK_FEATURES])[0])
    proj_epa  = float(epa_model.predict(X_epa[EPA_FEATURES])[0])

    # Cluster
    X_clust = pd.DataFrame([{f: row.get(f) for f in CLUSTER_FEATURES}]).fillna(
        pd.Series({f: row.get(f, 0) for f in CLUSTER_FEATURES}).median()
    )
    cluster_id = int(km.predict(km_scaler.transform(X_clust))[0])
    archetype  = archetype_names.get(cluster_id, "Unknown")

    return jsonify({
        "ml_prob":           round(prob, 4),
        "ml_projected_pick": proj_pick,
        "ml_projected_epa":  round(proj_epa, 1),
        "archetype_id":      cluster_id,
        "archetype":         archetype,
        "tier":              "Elite" if prob > 0.6 else "Starter" if prob > 0.4 else "Fringe" if prob > 0.25 else "Bust Risk",
    })


# ── /api/archetypes — cluster summary ───────────────────────────────────────

@app.route("/api/archetypes")
def get_archetypes():
    summary = {}
    for p in PLAYERS:
        aid = p.get("archetype_id")
        if aid is None:
            continue
        aid = int(aid)
        if aid not in summary:
            summary[aid] = {
                "id":       aid,
                "name":     archetype_names.get(aid, "Unknown"),
                "players":  0,
                "successes": 0,
                "avg_prob": 0,
                "avg_epa":  0,
            }
        summary[aid]["players"] += 1
        if p.get("success") == 1:
            summary[aid]["successes"] += 1
        summary[aid]["avg_prob"] += p.get("ml_prob") or 0
        summary[aid]["avg_epa"]  += p.get("career_epa") or 0

    for aid, s in summary.items():
        n = s["players"]
        s["success_rate"] = round(s["successes"] / n, 3) if n else 0
        s["avg_prob"]     = round(s["avg_prob"] / n, 3)  if n else 0
        s["avg_epa"]      = round(s["avg_epa"]  / n, 1)  if n else 0

    return jsonify(list(summary.values()))


if __name__ == "__main__":
    print("\n🏈  TE Draft Intelligence API")
    print("  → http://localhost:5050/api/status\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
