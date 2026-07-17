"""
PLE Readiness Recommendation Engine

Provides:
1. Weak subject identification
2. Personalized intervention recommendation
3. Study hour recommendation
4. Risk explanation
"""

from __future__ import annotations

import pandas as pd

# ----------------------------------------------------
# Weekly Study Hour Recommendation
# ----------------------------------------------------

DEFAULT_STUDY_HOURS = {
    "Ready":
        "Maintain 6–8 focused study hours per week. Continue answering high-yield board-style questions and reinforce strengths.",
    "Moderately Ready":
        "Schedule 10–12 focused study hours per week. Prioritize weak subjects and complete weekly self-assessments.",
    "At Risk":
        "Schedule at least 15 focused study hours per week with faculty coaching, structured review sessions, and weekly reassessment."
}


# ----------------------------------------------------
# Weak Subject Identification
# ----------------------------------------------------

def identify_weak_subjects(subject_scores: pd.Series,
                           top_n: int = 3) -> list[str]:
    """
    Returns the lowest-performing subject areas.
    """

    numeric = pd.to_numeric(subject_scores,
                            errors="coerce").dropna()

    return numeric.sort_values().head(top_n).index.tolist()


# ----------------------------------------------------
# Personalized Recommendation
# ----------------------------------------------------

def intervention_recommendation(readiness_level: str,
                                weak_subjects: list[str]) -> str:

    if weak_subjects:
        subject_text = ", ".join(weak_subjects)
    else:
        subject_text = "identified weak subject areas"

    study_plan = DEFAULT_STUDY_HOURS.get(
        readiness_level,
        DEFAULT_STUDY_HOURS["Moderately Ready"]
    )

    recommendation = f"""
Focus Areas:
• {subject_text}

Recommended Action:
• {study_plan}

Suggested Learning Strategies:
• Complete board-style practice questions.
• Review rationales for every incorrect answer.
• Rewatch lecture videos covering weak topics.
• Complete one mock examination every week.
• Track progress using the readiness dashboard.

Follow-up:
• Repeat readiness assessment after every major examination.
"""

    return recommendation.strip()