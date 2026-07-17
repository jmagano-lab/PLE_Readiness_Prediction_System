import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from src.recommendation_engine import (
    identify_weak_subjects,
    intervention_recommendation
)


BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "ple_readiness_cleaned.csv"
MODEL = BASE / "models" / "best_readiness_model.joblib"
FEATURES = ["diagnostic", "pre_test", "post_test", "progress_evaluation", "mock_boards", "early_score", "later_score", "improvement_index", "mock_to_diagnostic_gap", "assessment_consistency"]

st.set_page_config(page_title="PLE Readiness Prediction System", layout="wide")
st.title("PLE Readiness Prediction System")
st.caption("AI-powered readiness classification using diagnostic, pre-test, post-test, progress evaluation, and mock board scores.")

df = pd.read_csv(DATA)
model = joblib.load(MODEL)

c1, c2, c3 = st.columns(3)
c1.metric("Students", len(df))
c2.metric("Ready", int((df["readiness_level"] == "Ready").sum()))
c3.metric("At Risk", int((df["readiness_level"] == "At Risk").sum()))

st.subheader("Class Distribution")
st.bar_chart(df["readiness_level"].value_counts())

st.subheader("Student Prediction Simulator")
col1, col2, col3, col4, col5 = st.columns(5)
diagnostic = col1.number_input("Diagnostic", 0.0, 100.0, 10.0)
pre_test = col2.number_input("Pre-Test", 0.0, 100.0, 10.0)
post_test = col3.number_input("Post-Test", 0.0, 100.0, 1.0)
progress = col4.number_input("Progress Eval", 0.0, 100.0, 10.0)
mock = col5.number_input("Mock Boards", 0.0, 100.0, 20.0)

sample = pd.DataFrame([{
    "diagnostic": diagnostic,
    "pre_test": pre_test,
    "post_test": post_test,
    "progress_evaluation": progress,
    "mock_boards": mock,
}])
sample["early_score"] = sample[["diagnostic", "pre_test"]].mean(axis=1)
sample["later_score"] = sample[["post_test", "progress_evaluation", "mock_boards"]].mean(axis=1)
sample["improvement_index"] = sample["later_score"] - sample["early_score"]
sample["mock_to_diagnostic_gap"] = sample["mock_boards"] - sample["diagnostic"]
sample["assessment_consistency"] = sample[["diagnostic", "pre_test", "post_test", "progress_evaluation", "mock_boards"]].std(axis=1)

if st.button("Predict Readiness"):

    # Generate readiness prediction
    pred = model.predict(sample[FEATURES])[0]

    # Generate class probabilities
    proba = pd.DataFrame(
        model.predict_proba(sample[FEATURES]),
        columns=model.classes_
    ).T.rename(columns={0: "Probability"})

    # Display prediction
    st.success(f"Predicted Readiness Level: {pred}")

    # Display probability for each readiness category
    st.subheader("Prediction Probability")
    st.dataframe(
        (proba * 100).round(1).astype(str) + "%",
        use_container_width=True
    )

    # -------------------------------------------------
    # Weak-Area Identification
    # -------------------------------------------------

    assessment_scores = pd.Series({
        "Diagnostic Foundation": diagnostic,
        "Progress Evaluation": progress,
        "Post-Test Performance": post_test,
        "Masterclass Performance": pre_test,
        "Mock Board Performance": mock
    })

    weak_areas = identify_weak_subjects(
        assessment_scores,
        top_n=3
    )

    # Keep only areas with scores below 75
    weak_areas = [
        area for area in weak_areas
        if assessment_scores[area] < 75
    ]

    # -------------------------------------------------
    # Intervention Recommendation
    # -------------------------------------------------

    recommendation = intervention_recommendation(
        readiness_level=pred,
        weak_subjects=weak_areas
    )

    # Display weak areas
    st.subheader("Priority Areas for Improvement")

    if weak_areas:
        for area in weak_areas:
            st.write(
                f"• {area}: {assessment_scores[area]:.1f}%"
            )
    else:
        st.write(
            "No major weak assessment area was identified. "
            "Maintain current performance and continue high-yield review."
        )

    # Display personalized recommendation
    st.subheader("Personalized Intervention Recommendation")

    if pred == "Ready":
        st.success(recommendation)

    elif pred == "Moderately Ready":
        st.warning(recommendation)

    else:
        st.error(recommendation)


st.subheader("Student-Level Data")
st.dataframe(df, use_container_width=True)



