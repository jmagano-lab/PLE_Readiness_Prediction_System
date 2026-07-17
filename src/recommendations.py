"""Weak-area identification and intervention recommendations for PLE readiness."""
from __future__ import annotations
import pandas as pd

DEFAULT_STUDY_HOURS = {
    "Ready": "Maintain 6-8 focused hours/week with high-yield question practice.",
    "Moderately Ready": "Schedule 10-12 focused hours/week with targeted remediation.",
    "At Risk": "Schedule at least 15 focused hours/week plus faculty coaching and weekly reassessment.",
}

def identify_weak_subjects(subject_scores: pd.Series, top_n: int = 3) -> list[str]:
    """Return the top_n weakest subjects from a student's subject-level scores."""
    numeric = pd.to_numeric(subject_scores, errors="coerce").dropna()
    return numeric.sort_values().head(top_n).index.tolist()


def intervention_recommendation(readiness_level: str, weak_subjects: list[str]) -> str:
    """Generate a concise intervention plan based on readiness and weak subjects."""
    subjects = ", ".join(weak_subjects) if weak_subjects else "identified weak areas"
    base = DEFAULT_STUDY_HOURS.get(readiness_level, DEFAULT_STUDY_HOURS["Moderately Ready"])
    return f"Focus on {subjects}. {base} Use practice questions, rationales, and follow-up progress checks."
