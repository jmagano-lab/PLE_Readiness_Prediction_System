# PLE Readiness Prediction System

An AI-powered educational data mining and learning analytics capstone project for predicting medical students' Physician Licensure Examination readiness using diagnostic, pre-test, post-test, progress evaluation, and mock board examination performance.

## P.S.

This project represents one of the most challenging and rewarding learning experiences I have undertaken.

I am a medical doctor by profession, not a software engineer or computer scientist. Before starting this capstone, I had little to no background in programming, machine learning, or software development. Throughout this project, I dedicated significant time to learning Python, data science, machine learning concepts, and application development in order to transform an idea into a working system.

While I utilized modern AI-assisted development tools to help me understand concepts, and improve documentation, every component of this project was carefully studied, tested, modified, validated, and integrated by me. This capstone reflects my commitment to continuous learning and demonstrates that professionals from non-computer science backgrounds can successfully build practical AI solutions through perseverance, curiosity, and disciplined study.

Dataset Availability

The dataset used in this project is intentionally not included in this repository.

The data were provided by PLE REAP Medicine under a confidentiality agreement and consist of records from the 2025 batch of reviewees. To protect the privacy of the students and comply with the agreed data-sharing conditions, the original dataset cannot be publicly distributed.

For this reason:

The repository does not include the original dataset.
No personally identifiable student information is shared in this repository.
Anyone wishing to reproduce or extend this research should use their own appropriately collected and ethically approved dataset or obtain the necessary authorization from the data owner.

Thank you for taking the time to review this project. I hope it serves not only as a machine learning application for medical education but also as an example that determination and lifelong learning can bridge disciplines and lead to meaningful innovation.


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


