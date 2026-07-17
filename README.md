# PLE Readiness Prediction System

An AI-powered educational data mining and learning analytics capstone project for predicting medical students' Physician Licensure Examination readiness using diagnostic, pre-test, post-test, progress evaluation, and mock board examination performance.

## Dataset
- Source workbook: `data/source_uploaded_dataset.xlsx`
- Cleaned modeling dataset: `data/ple_readiness_cleaned.csv`
- Students analyzed: 229
- Target label: `readiness_level` with thresholds: Ready >=75, Moderately Ready 50-74, At Risk <50.

## Best Model
- Selected model: Random Forest
- Holdout accuracy: 1.000
- Holdout macro F1: 1.000

Important: high scores are expected because the readiness label is derived from component scores in the uploaded dataset. For real deployment, replace the target with actual PLE pass/fail or validated readiness outcomes.

## How to Run
```bash
pip install -r requirements.txt
python src/train_model.py
streamlit run dashboard_app.py
```

## Project Contents
- `reports/PLE_Readiness_Capstone_Report.docx` — full written capstone report
- `slides/PLE_Readiness_Technical_Presentation.pptx` — 8-12 slide technical deck
- `slides/PLE_Readiness_Business_Presentation.pptx` — 8-12 slide business deck
- `notebooks/PLE_Readiness_Modeling_Notebook.ipynb` — reproducible notebook
- `src/train_model.py` — documented model training pipeline
- `src/recommendations.py` — weak-area and intervention recommendation functions
- `dashboard_app.py` — Streamlit dashboard prototype
- `models/best_readiness_model.joblib` — trained model artifact
