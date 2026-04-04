"""
Simulated student engagement dataset for a university monitoring prototype (dissertation demo).

All records are synthetic. No real student data is used.
Single entry point: generates CSVs, summary markdown, charts, and matches DATA_GENERATION_INSTRUCTIONS.md.

Run: python generate_student_engagement_dataset.py
"""

from __future__ import annotations

import re
import random
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from faker import Faker

import sys

# Same risk implementation as risk-scoring-engine/risk_scoring_engine.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "risk-scoring-engine"))
from risk_scoring_engine import (  # noqa: E402
    composite_risk_pre_from_row,
    compute_risk_from_student_row,
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
Faker.seed(RANDOM_SEED)
fake = Faker("en_GB")
rng = np.random.default_rng(RANDOM_SEED)

# Demo "as of" date for summaries (e.g. alerts in the last 7 days)
AS_OF_DATE = datetime(2025, 3, 22).date()

# Paths: script lives in data-generation/; CSVs and charts are under project root.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
CHARTS_DIR = PROJECT_ROOT / "charts"

PROGRAMS = [
    "Computer Science",
    "Data Science",
    "Business Analytics",
    "Engineering",
    "Psychology",
]
FINANCIAL_SUPPORT = ["Full", "Partial", "None"]

N_STUDENTS = 500


def slug_email_part(name: str) -> str:
    """Lowercase ASCII-ish local part for email (no spaces/specials)."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z]+", "", s)
    return s or "student"


def generate_students(n: int = N_STUDENTS) -> pd.DataFrame:
    """
    Build correlated engagement metrics with intentional cohorts:
    - ~20% at-risk cluster (low latent engagement)
    - ~5% resilient (60–70% attendance but strong grades)
    - Commuters: slightly lower attendance
    - Part-time: slightly lower VLE time
    """
    rows = []

    # Mutually exclusive special cohorts for cleaner storytelling
    idx_all = np.arange(n)
    rng.shuffle(idx_all)
    resilient_idx = set(idx_all[: max(1, int(round(0.05 * n)))])
    remaining = [i for i in idx_all if i not in resilient_idx]
    at_risk_idx = set(remaining[: max(1, int(round(0.20 * n)))])

    for i in range(n):
        student_num = i + 1
        student_id = f"STU{student_num:04d}"

        first = fake.first_name()
        last = fake.last_name()
        student_name = f"{first} {last}"
        local_a = slug_email_part(first)
        local_b = slug_email_part(last)
        email = f"{local_a}.{local_b}@student.chester.ac.uk"

        program = rng.choice(PROGRAMS)
        year_of_study = int(rng.integers(1, 4))
        # Enrollment: September or October 2024
        month = int(rng.choice([9, 10]))
        # September: 30 days; October: 31 days
        day = int(rng.integers(1, 31)) if month == 9 else int(rng.integers(1, 32))
        enrollment_date = datetime(2024, month, day).date()

        is_commuter = rng.random() < 0.30
        works_part_time = rng.random() < 0.40
        has_declared_disability = rng.random() < 0.15
        is_international = rng.random() < 0.25
        financial_support = rng.choice(FINANCIAL_SUPPORT)

        # Latent engagement factor drives correlations (imperfect: noise added later)
        # Latent engagement: lowered tail for ~20% cluster, but tuned so rule-based
        # risk_score does not push half the cohort into "High" when rules stack.
        latent = float(rng.normal(0.26, 0.88))
        if i in at_risk_idx:
            latent -= 0.82
        if i in resilient_idx:
            latent += 0.45  # slight boost for grades; attendance set separately

        commuter_penalty = rng.normal(3.2, 1.4) if is_commuter else 0.0
        attendance_base = 77.0 + 10.5 * latent - commuter_penalty + rng.normal(0, 7.0)
        if i in resilient_idx:
            attendance_rate = float(np.clip(rng.uniform(60.0, 70.0), 0, 100))
        else:
            attendance_rate = float(np.clip(attendance_base, 0, 100))

        # VLE logins: tied to attendance + latent, with noise (0–25)
        login_mu = 4.4 + 0.11 * (attendance_rate - 77.0) + 4.6 * latent
        login_mu = np.clip(login_mu, 0.5, 24.5)
        vle_logins_last_week = int(np.clip(rng.poisson(login_mu), 0, 25))

        # Hours: scales with logins; part-time students have less spare time online
        time_base = 0.62 * vle_logins_last_week + rng.normal(0, 2.0)
        if works_part_time:
            time_base -= rng.uniform(1.0, 3.0)
        vle_time_hours_last_week = float(np.clip(time_base, 0.0, 20.0))
        # If they logged in at all, assume at least a few minutes online (noise can otherwise clip to 0).
        if vle_logins_last_week > 0:
            floor_h = float(rng.uniform(0.2, 1.0) * min(4, vle_logins_last_week))
            vle_time_hours_last_week = float(min(20.0, max(vle_time_hours_last_week, floor_h)))

        # Assignments (8 total): on-time vs late, correlated with engagement
        p_late = float(
            np.clip(0.08 - 0.02 * latent + 0.003 * max(0, 70 - attendance_rate), 0.02, 0.55)
        )
        late = int(rng.binomial(3, min(p_late * 1.2, 0.95)))
        late = min(late, 3)
        max_on_time = 8 - late
        on_time = int(
            rng.binomial(
                n=max_on_time,
                p=float(np.clip(0.55 + 0.15 * latent + 0.002 * (attendance_rate - 70), 0.05, 0.98)),
            )
        )
        on_time = min(on_time, 8)
        # Keep total submitted <= 8
        total_sub = on_time + late
        if total_sub > 8:
            late = max(0, 8 - on_time)

        # Grades: correlate with latent + attendance + VLE; resilient students buck the trend
        grade_mean = 66 + 11 * latent + 0.15 * (attendance_rate - 77) + 0.8 * (vle_logins_last_week - 8)
        if i in resilient_idx:
            grade_mean += rng.uniform(14, 24)
        last_assessment_grade = int(np.clip(rng.normal(grade_mean, 10.0), 0, 100))

        # Previous GPA correlated with current performance (not identical)
        gpa_mu = 1.25 + 0.021 * last_assessment_grade + 0.11 * latent
        previous_semester_gpa = float(np.clip(rng.normal(gpa_mu, 0.35), 0.0, 4.0))

        # Synthetic engagement trend (latent-driven; no circular use of risk score)
        historical_proxy = (
            0.45 * (attendance_rate / 100)
            + 0.25 * (previous_semester_gpa / 4.0)
            + 0.20 * (last_assessment_grade / 100)
            + 0.10 * (vle_logins_last_week / 25)
        )
        recent_shift = rng.normal(0, 0.065)
        if latent < 0.15:
            recent_shift -= rng.uniform(0.03, 0.14)
        elif latent > 0.65:
            recent_shift += rng.uniform(0.02, 0.12)
        recent_proxy = float(np.clip(historical_proxy + recent_shift, 0, 1))
        delta = recent_proxy - historical_proxy
        if delta > 0.02:
            engagement_trend = "Improving"
        elif delta < -0.02:
            engagement_trend = "Declining"
        else:
            engagement_trend = "Stable"

        # Upcoming assessment prep (synthetic student-level; supports "monitoring opportunities")
        p_access = float(
            np.clip(0.12 + 0.48 * latent + 0.0035 * (attendance_rate - 72.0), 0.04, 0.97)
        )
        accessed_upcoming_assessment_brief = bool(rng.random() < p_access)
        days_to_nearest_assessment = int(rng.integers(3, 22))

        rows.append(
            {
                "student_id": student_id,
                "student_name": student_name,
                "email": email,
                "program": program,
                "year_of_study": year_of_study,
                "enrollment_date": enrollment_date.isoformat(),
                "attendance_rate": round(attendance_rate, 2),
                "vle_logins_last_week": vle_logins_last_week,
                "vle_time_hours_last_week": round(vle_time_hours_last_week, 2),
                "assignments_submitted_on_time": on_time,
                "assignments_submitted_late": late,
                "last_assessment_grade": last_assessment_grade,
                "previous_semester_gpa": round(previous_semester_gpa, 2),
                "is_commuter": is_commuter,
                "works_part_time": works_part_time,
                "has_declared_disability": has_declared_disability,
                "is_international": is_international,
                "financial_support": financial_support,
                "engagement_trend": engagement_trend,
                "accessed_upcoming_assessment_brief": accessed_upcoming_assessment_brief,
                "days_to_nearest_assessment": days_to_nearest_assessment,
                "last_alert_date": pd.NaT,
                "alert_response_status": np.nan,
            }
        )

    df = pd.DataFrame(rows)
    df["risk_score_pre_intervention"] = df.apply(
        lambda r: composite_risk_pre_from_row(r.to_dict())[0], axis=1
    )
    df["last_alert_date"] = ""
    df["alert_response_status"] = ""
    return df


def finalize_risk_columns(
    students: pd.DataFrame, interventions: pd.DataFrame
) -> pd.DataFrame:
    """Merge intervention history, compute composite v2 risk, assign alerts."""
    s = students.copy()
    if len(interventions) > 0:
        agg = interventions.groupby("student_id").agg(
            prior_intervention_count=("intervention_id", "count"),
            prior_interventions_improved=("engagement_improved", "sum"),
        ).reset_index()
        s = s.merge(agg, on="student_id", how="left")
    else:
        s["prior_intervention_count"] = 0
        s["prior_interventions_improved"] = 0
    s["prior_intervention_count"] = s["prior_intervention_count"].fillna(0).astype(int)
    s["prior_interventions_improved"] = s["prior_interventions_improved"].fillna(0).astype(int)

    out_scores: list[int] = []
    out_levels: list[str] = []
    out_iv: list[float] = []
    for _, r in s.iterrows():
        o = compute_risk_from_student_row(r.to_dict())
        out_scores.append(o["risk_score"])
        out_levels.append(o["risk_level"])
        out_iv.append(float(o["intervention_history_stress"]))
    s["risk_score"] = out_scores
    s["risk_level"] = out_levels
    s["intervention_history_stress"] = out_iv

    last_dates: list[str] = []
    alert_status: list[str] = []
    for _i in range(len(s)):
        if rng.random() < 0.30:
            d = AS_OF_DATE - timedelta(days=int(rng.integers(0, 45)))
            last_dates.append(d.isoformat())
            alert_status.append(
                str(
                    rng.choice(
                        ["Responded", "No Response", "Meeting Scheduled"],
                        p=[0.45, 0.35, 0.20],
                    )
                )
            )
        else:
            last_dates.append("")
            alert_status.append("")
    s["last_alert_date"] = last_dates
    s["alert_response_status"] = alert_status
    return s


def generate_assessments(n: int | None = None) -> pd.DataFrame:
    """Upcoming assessments spread across the next 8 weeks from AS_OF_DATE."""
    if n is None:
        n = int(rng.integers(30, 41))  # 30–40 inclusive upper exclusive -> 41 gives up to 40

    prefixes = ["CS", "DA", "BA", "EN", "PS"]
    titles_pool = [
        "Research Methods Essay",
        "Data Analysis Project",
        "Statistics Midterm Exam",
        "Group Presentation",
        "Literature Review",
        "Programming Coursework",
        "Case Study Report",
        "Lab Practical Assessment",
        "Final Examination",
        "Reflective Portfolio",
        "Research Proposal",
        "Team Project Deliverable",
    ]
    types = ["Essay", "Project", "Exam", "Presentation"]
    weights = [10, 15, 20, 30, 40, 50]

    rows = []
    start = AS_OF_DATE
    for k in range(n):
        aid = f"ASS{k + 1:04d}"
        code = f"{rng.choice(prefixes)}{rng.integers(101, 401)}"
        title = f"{rng.choice(titles_pool)}"
        due = start + timedelta(days=int(rng.integers(1, 57)))  # spread inside ~8 weeks
        a_type = rng.choice(types)
        weight = int(rng.choice(weights))
        enrolled = int(rng.integers(40, 181))
        accessed = int(rng.binomial(enrolled, p=float(rng.uniform(0.35, 0.92))))
        not_accessed = enrolled - accessed
        rows.append(
            {
                "assessment_id": aid,
                "module_code": code,
                "assessment_title": title,
                "due_date": due.isoformat(),
                "assessment_type": a_type,
                "weight_percentage": weight,
                "students_enrolled": enrolled,
                "students_accessed_brief": accessed,
                "students_not_accessed": not_accessed,
            }
        )
    return pd.DataFrame(rows)


def generate_interventions(students: pd.DataFrame, n: int | None = None) -> pd.DataFrame:
    """Historical interventions linked to student_id (biased toward higher pre-risk)."""
    if n is None:
        n = int(rng.integers(200, 301))

    student_ids = students["student_id"].tolist()
    risky_ids = students.loc[
        students["risk_score_pre_intervention"] >= 52, "student_id"
    ].tolist()
    types = ["Email", "SMS", "Phone Call", "In-Person Meeting"]
    reasons = [
        "Low Attendance",
        "Missed Deadline",
        "Declining Grades",
        "No VLE Activity",
    ]

    rows = []
    for k in range(n):
        if len(risky_ids) and rng.random() < 0.62:
            sid = str(rng.choice(risky_ids))
        else:
            sid = str(rng.choice(student_ids))
        # Last ~90 days
        intervention_date = AS_OF_DATE - timedelta(days=int(rng.integers(0, 91)))
        itype = rng.choice(types, p=[0.40, 0.25, 0.20, 0.15])
        reason = rng.choice(reasons)
        responded = bool(rng.random() < 0.55)

        if responded:
            response_time_hours = int(rng.integers(0, 169))
            improved = bool(rng.random() < 0.70)
        else:
            response_time_hours = np.nan
            improved = bool(rng.random() < 0.20)

        row_student = students.loc[students["student_id"] == sid].iloc[0]
        att_base = float(row_student["attendance_rate"])
        grade_base = int(row_student["last_assessment_grade"])

        if improved:
            attendance_after = float(np.clip(att_base + rng.uniform(3, 18), 0, 100))
            grade_after = int(np.clip(grade_base + rng.integers(3, 22), 0, 100))
        else:
            attendance_after = float(np.clip(att_base + rng.normal(0, 4), 0, 100))
            grade_after = int(np.clip(grade_base + rng.integers(-8, 9), 0, 100))

        rows.append(
            {
                "intervention_id": f"INT{k + 1:04d}",
                "student_id": sid,
                "intervention_date": intervention_date.isoformat(),
                "intervention_type": itype,
                "reason": reason,
                "student_responded": responded,
                "response_time_hours": int(response_time_hours) if responded else "",
                "engagement_improved": improved,
                "attendance_before": round(att_base, 2),
                "attendance_after": round(attendance_after, 2),
                "grade_before": grade_base,
                "grade_after": grade_after,
            }
        )
    return pd.DataFrame(rows)


def write_dataset_summary(students: pd.DataFrame, interventions: pd.DataFrame) -> Path:
    """Submission-ready Markdown summary."""
    last_week_start = AS_OF_DATE - timedelta(days=7)
    students_work = students.copy()
    students_work["last_alert_date_parsed"] = pd.to_datetime(
        students_work["last_alert_date"], errors="coerce"
    )
    flagged_last_week = students_work["last_alert_date_parsed"].apply(
        lambda d: pd.notna(d) and (d.date() >= last_week_start) and (d.date() <= AS_OF_DATE)
    ).sum()

    by_risk = students.groupby("risk_level")
    avg_att = by_risk["attendance_rate"].mean().round(2)
    avg_grade = by_risk["last_assessment_grade"].mean().round(2)

    inter = interventions.copy()
    inter["student_responded"] = inter["student_responded"].astype(bool)
    success_rate = 100.0 * inter["engagement_improved"].mean()

    def channel_response_rate(itype: str) -> float:
        sub = inter[inter["intervention_type"] == itype]
        if len(sub) == 0:
            return float("nan")
        return 100.0 * sub["student_responded"].mean()

    email_rr = channel_response_rate("Email")
    sms_rr = channel_response_rate("SMS")
    phone_rr = channel_response_rate("Phone Call")

    risk_counts = students["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).astype(int)

    lines = [
        "# Simulated student engagement dataset — summary",
        "",
        f"_Generated for demonstration purposes only. As-of date used in summaries: **{AS_OF_DATE.isoformat()}**._",
        "",
        "## Overview",
        "",
        f"- **Total students:** {len(students)}",
        f"- **Risk level — Low:** {risk_counts.get('Low', 0)}",
        f"- **Risk level — Medium:** {risk_counts.get('Medium', 0)}",
        f"- **Risk level — High:** {risk_counts.get('High', 0)}",
        "",
        "## Average attendance and grades by risk level",
        "",
        "| Risk level | Avg attendance (%) | Avg last assessment grade |",
        "|------------|-------------------:|--------------------------:|",
    ]
    for level in ["Low", "Medium", "High"]:
        if level in avg_att.index:
            lines.append(
                f"| {level} | {avg_att[level]} | {avg_grade[level]} |"
            )
        else:
            lines.append(f"| {level} | — | — |")

    lines.extend(
        [
            "",
            "## Alerts",
            "",
            f"- **Students with an alert dated in the last 7 days (inclusive):** {int(flagged_last_week)}",
            "",
            "## Interventions (historical sample)",
            "",
            f"- **Total intervention records:** {len(inter)}",
            f"- **Engagement improved (any intervention):** {success_rate:.1f}%",
            f"- **Response rate — Email:** {email_rr:.1f}%",
            f"- **Response rate — SMS:** {sms_rr:.1f}%",
            f"- **Response rate — Phone Call:** {phone_rr:.1f}%",
            "",
            "## How to interpret risk scores",
            "",
            "**Composite model (v2):** weighted blend of eight 0–100 stress components",
            "(attendance, VLE logins, VLE hours, grade, submission behaviour, trajectory,",
            "GPA gap, assessment preparation), with contextual easing for commuters,",
            "part-time workers, and declared disability, then blended with",
            "**intervention history stress** derived from past outreach counts and outcomes.",
            "",
            "**Risk level bands:** Low 0–33, Medium 34–66, High 67–100.",
            "",
            "---",
            "",
            "_This file is produced automatically by `generate_student_engagement_dataset.py`._",
        ]
    )

    out_path = SCRIPT_DIR / "dataset_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def save_charts(students: pd.DataFrame, interventions: pd.DataFrame) -> list[Path]:
    """Create five PNG charts in CHARTS_DIR."""
    saved: list[Path] = []
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    # 1) Risk score histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(students["risk_score"], bins=20, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Risk score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of risk scores (simulated)")
    fig.tight_layout()
    p1 = CHARTS_DIR / "risk_score_distribution.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saved.append(p1)

    # 2) Attendance vs grade, colored by risk
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Low": "#55A868", "Medium": "#FCC44C", "High": "#C44E52"}
    for level in ["Low", "Medium", "High"]:
        sub = students[students["risk_level"] == level]
        ax.scatter(
            sub["attendance_rate"],
            sub["last_assessment_grade"],
            s=22,
            alpha=0.75,
            label=level,
            c=colors[level],
            edgecolors="none",
        )
    ax.set_xlabel("Attendance rate (%)")
    ax.set_ylabel("Last assessment grade")
    ax.set_title("Attendance vs grade by risk level")
    ax.legend(title="Risk level")
    fig.tight_layout()
    p2 = CHARTS_DIR / "attendance_vs_grade_scatter.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    saved.append(p2)

    # 3) VLE logins vs grade
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        students["vle_logins_last_week"],
        students["last_assessment_grade"],
        c=students["risk_score"],
        cmap="viridis",
        alpha=0.8,
        s=28,
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Risk score")
    ax.set_xlabel("VLE logins (last week)")
    ax.set_ylabel("Last assessment grade")
    ax.set_title("VLE activity vs grade (colour = risk score)")
    fig.tight_layout()
    p3 = CHARTS_DIR / "vle_vs_grade_scatter.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    saved.append(p3)

    # 4) Intervention response rates by channel (Email, SMS, Phone)
    fig, ax = plt.subplots(figsize=(8, 5))
    channels = ["Email", "SMS", "Phone Call"]
    rates = []
    for c in channels:
        sub = interventions[interventions["intervention_type"] == c]
        rates.append(100.0 * sub["student_responded"].mean() if len(sub) else 0.0)
    x = np.arange(len(channels))
    ax.bar(x, rates, color=["#4C72B0", "#8172B2", "#CCB974"])
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.set_ylabel("Response rate (%)")
    ax.set_title("Student response rate by intervention channel")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    p4 = CHARTS_DIR / "intervention_response_rates.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    saved.append(p4)

    # 5) Engagement trend pie
    fig, ax = plt.subplots(figsize=(6, 6))
    trend_counts = students["engagement_trend"].value_counts().reindex(
        ["Improving", "Stable", "Declining"]
    )
    trend_counts = trend_counts.fillna(0).astype(int)
    ax.pie(
        trend_counts.values,
        labels=trend_counts.index,
        autopct="%1.1f%%",
        colors=["#55A868", "#DDDDDD", "#C44E52"],
        startangle=90,
    )
    ax.set_title("Engagement trend (simulated proxy)")
    fig.tight_layout()
    p5 = CHARTS_DIR / "engagement_trend_pie.png"
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    saved.append(p5)

    return saved


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating students...")
    students = generate_students(N_STUDENTS)

    print("Generating assessments...")
    assessments = generate_assessments()
    assessments_path = DATA_DIR / "assessments_data.csv"
    assessments.to_csv(assessments_path, index=False)
    print(f"Wrote {assessments_path} ({len(assessments)} rows)")

    print("Generating interventions...")
    interventions = generate_interventions(students)
    interventions_path = DATA_DIR / "interventions_data.csv"
    interventions.to_csv(interventions_path, index=False)
    print(f"Wrote {interventions_path} ({len(interventions)} rows)")

    print("Finalising composite risk scores (v2)...")
    students = finalize_risk_columns(students, interventions)
    students_path = DATA_DIR / "students_data.csv"
    students.to_csv(students_path, index=False)
    print(f"Wrote {students_path}")

    print("Writing dataset summary...")
    summary_path = write_dataset_summary(students, interventions)
    print(f"Wrote {summary_path}")

    print("Saving charts...")
    chart_paths = save_charts(students, interventions)
    for p in chart_paths:
        print(f"Wrote {p}")

    print("Done.")


if __name__ == "__main__":
    main()
