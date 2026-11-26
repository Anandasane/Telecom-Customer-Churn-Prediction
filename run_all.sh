#!/usr/bin/env bash
set -e
python -m src.eda
python -m src.train
python -m src.evaluate
python -m pytest -q tests
echo "Done. Launch dashboard with: streamlit run app/streamlit_app.py"
