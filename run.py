#!/usr/bin/env python3
"""
run.py  —  TE Draft Intelligence · One-command launcher

Usage:
    python run.py           # train models (if needed) then start API
    python run.py --train   # force retrain even if results already exist
    python run.py --api     # skip training, just start the API server

The API will be available at http://localhost:5050
"""

import argparse
import os
import sys

RESULTS_EXIST = (
    os.path.exists("results/players.json")
    and os.path.exists("results/prospects.json")
    and os.path.exists("models/ensemble.pkl")
)


def train():
    print("=" * 60)
    print("  Step 1 of 2 — Training ML models")
    print("=" * 60)
    from ml_models import main
    main()
    print("\n✓  Training complete.\n")


def serve():
    print("=" * 60)
    print("  Step 2 of 2 — Starting API server")
    print("  → http://localhost:5050/api/status")
    print("=" * 60)
    # Import and start the Flask app
    from api_server import app
    app.run(host="0.0.0.0", port=5050, debug=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TE Draft Intelligence")
    parser.add_argument("--train", action="store_true", help="Force retrain")
    parser.add_argument("--api",   action="store_true", help="API only (skip train)")
    args = parser.parse_args()

    if args.api:
        if not RESULTS_EXIST:
            print("ERROR: No results found. Run without --api first to train.")
            sys.exit(1)
        serve()
    else:
        if args.train or not RESULTS_EXIST:
            train()
        else:
            print("✓  Pre-trained results found. Use --train to retrain.\n")
        serve()
