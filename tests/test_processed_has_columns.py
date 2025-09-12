# tests/test_processed_has_columns.py
from pathlib import Path
import pandas as pd
CSV = Path(__file__).resolve().parents[1] / "assets/data/processed/medical_imaging_text_labeled.csv"

def test_minimal_columns_exist():
    df = pd.read_csv(CSV)
    must = {"event_text","label"}
    assert must.issubset(df.columns), f"Colonnes requises manquantes: {must - set(df.columns)}"
