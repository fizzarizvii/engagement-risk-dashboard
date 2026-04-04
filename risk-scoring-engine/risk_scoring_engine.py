"""
Student engagement risk scoring — single module (model + CLI).

Composite v2: multiple metrics are combined into one 0–100 risk score, then blended
with intervention-history stress. Same logic is imported by
``data-generation/generate_student_engagement_dataset.py`` to build the CSV.

SRS alignment (honest summary):
  • 2.3.1 Multi-dimensional composite: YES (weighted components).
  • Trajectory: PARTIAL — uses categorical ``engagement_trend``, not week-by-week attendance.
  • VLE: logins + hours this week only — not “15 last week vs 0 this week” (no two-week series).
  • Assessment prep: PARTIAL — one synthetic nearest deadline + brief access, not per-module rows.
  • Library / tutoring: NOT in synthetic data.
  • 2.3.2 Pattern matching: PARTIAL — intervention history stress only, not k-NN failure rates.
  • 2.3.3 Per-assessment scores: NOT implemented — one overall score per student.
  • 2.3.4 Context (commuter, part-time, disability, GPA vs grade): YES.
  • 2.3.5 Dynamic quarterly calibration: NOT implemented — fixed bands and calibration constant.

Simulated data only; not for real students.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

# =============================================================================
# MODEL (v2) — used by data generator and by CLI
# =============================================================================

# Component weights for pre-intervention composite (sum = 1.0)
W_ATT = 0.18
W_VLE_LOGIN = 0.14
W_VLE_HOURS = 0.12
W_GRADE = 0.18
W_SUBMISSION = 0.12
W_TREND = 0.10
W_GPA_GAP = 0.10
W_PREP = 0.06

W_PRE = 0.78
W_INTERVENTION = 0.22

RISK_PRE_CALIBRATION = 1.48

ATT_THRESHOLD = 70.0
VLE_LOGIN_THRESHOLD = 5
VLE_HOURS_THRESHOLD = 6.0
GRADE_THRESHOLD = 60.0
MAX_LATE = 3
MAX_ASSIGNMENTS = 8


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _risk_level_from_score(score: float) -> str:
    s = int(round(score))
    if s <= 33:
        return "Low"
    if s <= 66:
        return "Medium"
    return "High"


def component_stresses(row: dict[str, Any]) -> dict[str, float]:
    """Map CSV row to eight 0–100 stress subscores (higher = worse)."""
    att = float(row["attendance_rate"])
    logins = int(row["vle_logins_last_week"])
    hours = float(row["vle_time_hours_last_week"])
    grade = int(row["last_assessment_grade"])
    on_time = int(row["assignments_submitted_on_time"])
    late = int(row["assignments_submitted_late"])
    gpa = float(row["previous_semester_gpa"])
    trend = str(row["engagement_trend"])
    commuter = bool(row["is_commuter"])
    part_time = bool(row["works_part_time"])
    disability = bool(row["has_declared_disability"])

    if att < ATT_THRESHOLD:
        s_att = _clip01((ATT_THRESHOLD - att) / ATT_THRESHOLD)
    else:
        s_att = 0.0
    if commuter:
        s_att *= 0.72
    if disability:
        s_att *= 0.85

    if logins < VLE_LOGIN_THRESHOLD:
        s_vle = _clip01((VLE_LOGIN_THRESHOLD - logins) / VLE_LOGIN_THRESHOLD)
    else:
        s_vle = 0.0

    if hours < VLE_HOURS_THRESHOLD:
        s_hrs = _clip01((VLE_HOURS_THRESHOLD - hours) / VLE_HOURS_THRESHOLD)
    else:
        s_hrs = 0.0
    if part_time:
        s_hrs *= 0.78

    if grade < GRADE_THRESHOLD:
        s_gr = _clip01((GRADE_THRESHOLD - grade) / GRADE_THRESHOLD)
    else:
        s_gr = 0.0

    late_frac = late / float(MAX_LATE) if MAX_LATE else 0.0
    ontime_frac = on_time / float(MAX_ASSIGNMENTS)
    s_sub = _clip01(0.55 * late_frac + 0.45 * (1.0 - ontime_frac))

    trend_map = {"Declining": 0.82, "Stable": 0.42, "Improving": 0.10}
    s_tr = trend_map.get(trend, 0.35)

    expected = gpa * 25.0
    gap = expected - float(grade)
    if gap > 8.0:
        s_gap = _clip01(gap / 35.0)
    else:
        s_gap = 0.0

    accessed = bool(row.get("accessed_upcoming_assessment_brief", True))
    days_left = int(row.get("days_to_nearest_assessment", 99))
    if days_left <= 21 and not accessed:
        urgency = _clip01((21 - max(0, days_left)) / 21.0)
        s_prep = 0.55 + 0.45 * urgency
    elif days_left <= 21 and accessed:
        s_prep = 0.08
    else:
        s_prep = 0.05

    return {
        "attendance_stress": 100.0 * s_att,
        "vle_login_stress": 100.0 * s_vle,
        "vle_hours_stress": 100.0 * s_hrs,
        "grade_stress": 100.0 * s_gr,
        "submission_stress": 100.0 * s_sub,
        "trajectory_stress": 100.0 * s_tr,
        "gpa_gap_stress": 100.0 * s_gap,
        "assessment_prep_stress": 100.0 * s_prep,
    }


def composite_risk_pre_from_row(row: dict[str, Any]) -> tuple[int, dict[str, float]]:
    """Weighted composite 0–100 before intervention history."""
    c = component_stresses(row)
    raw = (
        W_ATT * c["attendance_stress"]
        + W_VLE_LOGIN * c["vle_login_stress"]
        + W_VLE_HOURS * c["vle_hours_stress"]
        + W_GRADE * c["grade_stress"]
        + W_SUBMISSION * c["submission_stress"]
        + W_TREND * c["trajectory_stress"]
        + W_GPA_GAP * c["gpa_gap_stress"]
        + W_PREP * c["assessment_prep_stress"]
    )
    disability = bool(row["has_declared_disability"])
    if disability:
        raw = max(0.0, raw - 4.5)
    raw = raw * RISK_PRE_CALIBRATION
    score = int(round(np.clip(raw, 0.0, 100.0)))
    c["composite_pre_intervention"] = float(score)
    return score, c


def intervention_history_stress(n_interventions: int, n_improved: int) -> float:
    if n_interventions <= 0:
        return 0.0
    fail_rate = (n_interventions - n_improved) / float(n_interventions)
    return float(min(100.0, n_interventions * 11.0 + fail_rate * 38.0))


def final_risk_score(risk_pre: int, intervention_stress: float) -> int:
    blended = W_PRE * risk_pre + W_INTERVENTION * intervention_stress
    return int(round(np.clip(blended, 0.0, 100.0)))


def compute_risk_from_student_row(
    row: dict[str, Any],
    n_prior_interventions: int = 0,
    n_prior_improved: int = 0,
) -> dict[str, Any]:
    if "prior_intervention_count" in row and pd.notna(row["prior_intervention_count"]):
        n_prior_interventions = int(row["prior_intervention_count"])
        n_prior_improved = int(row.get("prior_interventions_improved", 0) or 0)

    risk_pre, components = composite_risk_pre_from_row(row)
    iv_stress = intervention_history_stress(n_prior_interventions, n_prior_improved)
    final = final_risk_score(risk_pre, iv_stress)
    return {
        "risk_score": final,
        "risk_level": _risk_level_from_score(final),
        "risk_score_pre_intervention": risk_pre,
        "intervention_history_stress": round(iv_stress, 2),
        "components": components,
    }


def explain_components_for_display(row: dict[str, Any]) -> list[tuple[str, float]]:
    c = component_stresses(row)
    return [
        ("Attendance stress", c["attendance_stress"]),
        ("VLE login stress", c["vle_login_stress"]),
        ("VLE hours stress", c["vle_hours_stress"]),
        ("Grade stress", c["grade_stress"]),
        ("Submission stress", c["submission_stress"]),
        ("Trajectory stress", c["trajectory_stress"]),
        ("GPA gap stress", c["gpa_gap_stress"]),
        ("Assessment prep stress", c["assessment_prep_stress"]),
    ]


def compute_risk_from_dataframe_row(series: pd.Series) -> dict[str, Any]:
    return compute_risk_from_student_row(series.to_dict())


# =============================================================================
# CLI, validation, reports
# =============================================================================

ENGINE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_CSV = ENGINE_DIR.parent / "data" / "students_data.csv"

DEFAULTS_MINIMAL_SCORING: dict[str, Any] = {
    "vle_time_hours_last_week": 4.0,
    "assignments_submitted_on_time": 6,
    "previous_semester_gpa": 2.5,
    "engagement_trend": "Stable",
    "is_commuter": False,
    "works_part_time": False,
    "has_declared_disability": False,
    "accessed_upcoming_assessment_brief": True,
    "days_to_nearest_assessment": 14,
    "prior_intervention_count": 0,
    "prior_interventions_improved": 0,
}

REQUIRED_CSV_COLUMNS = [
    "student_id",
    "attendance_rate",
    "vle_logins_last_week",
    "vle_time_hours_last_week",
    "assignments_submitted_on_time",
    "assignments_submitted_late",
    "last_assessment_grade",
    "previous_semester_gpa",
    "is_commuter",
    "works_part_time",
    "has_declared_disability",
    "engagement_trend",
    "accessed_upcoming_assessment_brief",
    "days_to_nearest_assessment",
    "prior_intervention_count",
    "prior_interventions_improved",
]


def _merge_defaults(student_data: dict[str, Any]) -> dict[str, Any]:
    merged = {**DEFAULTS_MINIMAL_SCORING, **student_data}
    if "assignments_submitted_on_time" not in student_data and "assignments_submitted_late" in merged:
        late = int(merged["assignments_submitted_late"])
        merged["assignments_submitted_on_time"] = max(0, 8 - late)
    return merged


def calculate_risk_score(student_data: dict[str, Any]) -> dict[str, Any]:
    for key in (
        "attendance_rate",
        "vle_logins_last_week",
        "last_assessment_grade",
        "assignments_submitted_late",
    ):
        if key not in student_data:
            raise KeyError(f"student_data must include {key}")
    merged = _merge_defaults(student_data)
    return compute_risk_from_student_row(merged)


@dataclass
class ValidationMetrics:
    total_students: int
    exact_matches: int
    close_matches: int
    mismatches: int
    within_five: int
    mismatch_rows: list[dict[str, Any]] = field(default_factory=list)
    recalculated_scores: np.ndarray | None = None
    original_scores: np.ndarray | None = None


def validate_against_dataset(
    csv_path: str | Path, *, verbose: bool = True
) -> dict[str, Any]:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    n = len(df)
    diffs: list[int] = []
    recalc: list[int] = []
    mismatch_rows: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        d = row.to_dict()
        if d["accessed_upcoming_assessment_brief"] in ("True", "true", "1", 1):
            d["accessed_upcoming_assessment_brief"] = True
        elif d["accessed_upcoming_assessment_brief"] in ("False", "false", "0", 0):
            d["accessed_upcoming_assessment_brief"] = False
        out = compute_risk_from_student_row(d)
        stored = int(row["risk_score"])
        calc = int(out["risk_score"])
        recalc.append(calc)
        diff = calc - stored
        diffs.append(diff)

        if abs(diff) > 5:
            mismatch_rows.append(
                {
                    "student_id": row.get("student_id", idx),
                    "expected_stored": stored,
                    "calculated": calc,
                    "difference": diff,
                }
            )

        if verbose and (idx + 1) % 100 == 0:
            print(f"  ... processed {idx + 1}/{n} rows")

    diffs_arr = np.array(diffs, dtype=int)
    exact = int(np.sum(diffs_arr == 0))
    close = int(np.sum((diffs_arr != 0) & (np.abs(diffs_arr) <= 5)))
    bad = int(np.sum(np.abs(diffs_arr) > 5))
    within_five = int(np.sum(np.abs(diffs_arr) <= 5))

    metrics = ValidationMetrics(
        total_students=n,
        exact_matches=exact,
        close_matches=close,
        mismatches=bad,
        within_five=within_five,
        mismatch_rows=mismatch_rows,
        recalculated_scores=np.array(recalc, dtype=int),
        original_scores=df["risk_score"].astype(int).values,
    )

    out_df = df.copy()
    out_df["risk_score_recalc"] = recalc
    out_df["score_diff"] = diffs_arr

    return {
        "metrics": metrics,
        "dataframe": out_df,
        "exact_pct": 100.0 * exact / n if n else 0.0,
        "close_pct": 100.0 * close / n if n else 0.0,
        "mismatch_pct": 100.0 * bad / n if n else 0.0,
        "within_five_pct": 100.0 * within_five / n if n else 0.0,
        "overall_accuracy_pct": 100.0 * within_five / n if n else 0.0,
    }


def batch_score_students(students_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, st in enumerate(students_list):
        try:
            r = calculate_risk_score(st)
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Student index {i}: {e}") from e
        merged = _merge_defaults(st)
        merged["risk_score"] = r["risk_score"]
        merged["risk_level"] = r["risk_level"]
        merged["risk_score_pre_intervention"] = r["risk_score_pre_intervention"]
        merged["intervention_history_stress"] = r["intervention_history_stress"]
        merged["risk_components"] = r["components"]
        out.append(merged)
    return out


def _distribution_counts(scores: np.ndarray | pd.Series) -> dict[str, int]:
    s = np.asarray(scores).astype(int)
    low = int(np.sum(s <= 33))
    med = int(np.sum((s >= 34) & (s <= 66)))
    high = int(np.sum(s >= 67))
    return {"Low": low, "Medium": med, "High": high}


def generate_validation_report(csv_path: str | Path, output_path: str | Path) -> Path:
    result = validate_against_dataset(csv_path, verbose=False)
    metrics: ValidationMetrics = result["metrics"]
    path_in = Path(csv_path)
    path_out = Path(output_path)

    recalc = metrics.recalculated_scores
    orig = metrics.original_scores
    if recalc is None or orig is None:
        raise RuntimeError("Validation did not produce score arrays")

    dist_new = _distribution_counts(recalc)
    dist_old = _distribution_counts(orig)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "RISK SCORING ALGORITHM VALIDATION REPORT (composite v2)",
        "",
        f"Dataset: {path_in.as_posix()}",
        f"Validation Date: {ts}",
        f"Total Students: {metrics.total_students}",
        "",
        "ACCURACY METRICS:",
        "",
        f"Exact Matches: {metrics.exact_matches}/{metrics.total_students} ({result['exact_pct']:.1f}%)",
        f"Close Matches (within 5 points, excluding exact): {metrics.close_matches}/{metrics.total_students} ({result['close_pct']:.1f}%)",
        f"Within 5 points (including exact): {metrics.within_five}/{metrics.total_students} ({result['within_five_pct']:.1f}%)",
        f"Mismatches (difference > 5): {metrics.mismatches}/{metrics.total_students} ({result['mismatch_pct']:.1f}%)",
        "",
        f"Overall Accuracy (within 5 points): {result['overall_accuracy_pct']:.1f}%",
        "",
        "SCORE DISTRIBUTION (Recalculated):",
        "",
        f"Low Risk (0-33): {dist_new['Low']} students",
        f"Medium Risk (34-66): {dist_new['Medium']} students",
        f"High Risk (67-100): {dist_new['High']} students",
        "",
        "SCORE DISTRIBUTION (Original Dataset):",
        "",
        f"Low Risk: {dist_old['Low']} students",
        f"Medium Risk: {dist_old['Medium']} students",
        f"High Risk: {dist_old['High']} students",
        "",
        "MISMATCH DETAILS:",
        "",
    ]

    if not metrics.mismatch_rows:
        lines.append("None — all students within 5 points of stored scores.")
    else:
        for m in metrics.mismatch_rows:
            lines.append(
                f"  {m['student_id']}: stored={m['expected_stored']}, "
                f"calculated={m['calculated']}, diff={m['difference']:+d}"
            )

    lines.extend(
        [
            "",
            "CONCLUSION:",
            "",
            "Composite v2 blends multi-metric stresses with intervention history. "
            "Exact recalculation should match the stored CSV if the dataset was "
            "generated with the same script and seed.",
            "",
        ]
    )

    path_out.parent.mkdir(parents=True, exist_ok=True)
    path_out.write_text("\n".join(lines), encoding="utf-8")
    return path_out.resolve()


def _color_for_level(level: str) -> str:
    if level == "Low":
        return Fore.GREEN
    if level == "Medium":
        return Fore.YELLOW
    return Fore.RED


def _coerce_float(val: Any, name: str, low: float, high: float) -> float:
    x = float(val)
    if not (low <= x <= high):
        raise ValueError(f"{name} must be between {low} and {high}")
    return x


def _coerce_int(val: Any, name: str, low: int, high: int) -> int:
    x = int(round(float(val)))
    if not (low <= x <= high):
        raise ValueError(f"{name} must be between {low} and {high}")
    return x


def interactive_scoring() -> None:
    print(f"\n{Style.BRIGHT}=== Student Risk Scoring (composite v2) ==={Style.RESET_ALL}\n")

    while True:
        print("Enter fields (defaults in brackets where used if you press Enter):\n")
        try:
            att = _coerce_float(input("Attendance rate 0-100 [%]: ").strip(), "attendance", 0, 100)
            vle = _coerce_int(input("VLE logins last week [0-25]: ").strip(), "vle", 0, 25)
            hrs = _coerce_float(
                input("VLE hours last week [0-20] (default 4): ").strip() or "4",
                "hours",
                0,
                20,
            )
            grade = _coerce_int(input("Last assessment grade 0-100: ").strip(), "grade", 0, 100)
            late = _coerce_int(input("Late assignments 0-3: ").strip(), "late", 0, 3)
            on_time = _coerce_int(
                input("On-time assignments 0-8 (default 8-late): ").strip() or str(max(0, 8 - late)),
                "on_time",
                0,
                8,
            )
            gpa = _coerce_float(
                input("Previous semester GPA 0-4 (default 2.5): ").strip() or "2.5",
                "gpa",
                0,
                4,
            )
            trend = (
                input("Engagement trend (Improving/Stable/Declining) [Stable]: ").strip() or "Stable"
            )
            if trend not in ("Improving", "Stable", "Declining"):
                raise ValueError("Trend must be Improving, Stable, or Declining")
            comm = (input("Commuter y/n [n]: ").strip().lower() or "n") == "y"
            pt = (input("Part-time work y/n [n]: ").strip().lower() or "n") == "y"
            dis = (input("Declared disability y/n [n]: ").strip().lower() or "n") == "y"
            acc = (input("Accessed upcoming assessment brief y/n [y]: ").strip().lower() or "y") == "y"
            days = _coerce_int(
                input("Days to nearest assessment (3-21) [14]: ").strip() or "14",
                "days",
                1,
                60,
            )
            n_iv = _coerce_int(
                input("Prior intervention count [0]: ").strip() or "0",
                "n_iv",
                0,
                50,
            )
            n_imp = _coerce_int(
                input("Prior interventions with improved engagement [0]: ").strip() or "0",
                "n_imp",
                0,
                n_iv,
            )
        except ValueError as e:
            print(f"{Fore.RED}{e}{Style.RESET_ALL}\n")
            continue
        except EOFError:
            print()
            break

        data = {
            "attendance_rate": att,
            "vle_logins_last_week": vle,
            "vle_time_hours_last_week": hrs,
            "assignments_submitted_on_time": on_time,
            "assignments_submitted_late": late,
            "last_assessment_grade": grade,
            "previous_semester_gpa": gpa,
            "engagement_trend": trend,
            "is_commuter": comm,
            "works_part_time": pt,
            "has_declared_disability": dis,
            "accessed_upcoming_assessment_brief": acc,
            "days_to_nearest_assessment": days,
            "prior_intervention_count": n_iv,
            "prior_interventions_improved": n_imp,
        }

        r = calculate_risk_score(data)
        colour = _color_for_level(r["risk_level"])

        print(f"\n{Style.BRIGHT}RESULT{Style.RESET_ALL}")
        print(f"Risk score: {r['risk_score']} / 100  ({colour}{r['risk_level'].upper()}{Style.RESET_ALL})")
        print(f"Pre-intervention composite: {r['risk_score_pre_intervention']}")
        print(f"Intervention history stress: {r['intervention_history_stress']}")
        print("\nComponent stresses (0-100, higher=worse):")
        for name, val in explain_components_for_display(_merge_defaults(data)):
            print(f"  {name}: {val:.1f}")

        again = input("\nAnother (y/n)? ").strip().lower()
        if again != "y":
            break


def _menu() -> None:
    default_csv = DEFAULT_DATA_CSV
    report_path = ENGINE_DIR / "validation_report.txt"

    while True:
        print(f"\n{Style.BRIGHT}--- Risk Scoring Engine (v2) ---{Style.RESET_ALL}")
        print(f"Dataset: {default_csv.as_posix()}")
        print("1. Validate against dataset")
        print("2. Interactive scoring")
        print("3. Generate validation report")
        print("4. Exit")
        choice = input("\nSelect (1-4): ").strip()

        if choice == "1":
            if not default_csv.is_file():
                print(f"{Fore.RED}Missing: {default_csv}{Style.RESET_ALL}")
                continue
            print(f"\nLoading {default_csv}...\n")
            try:
                res = validate_against_dataset(default_csv)
            except Exception as e:
                print(f"{Fore.RED}{e}{Style.RESET_ALL}")
                continue
            m: ValidationMetrics = res["metrics"]
            print(f"Exact: {m.exact_matches}/{m.total_students} ({res['exact_pct']:.1f}%)")
            print(f"Within 5: {m.within_five}/{m.total_students} ({res['within_five_pct']:.1f}%)")
            print(f"Mismatches >5: {m.mismatches}")

        elif choice == "2":
            interactive_scoring()

        elif choice == "3":
            if not default_csv.is_file():
                print(f"{Fore.RED}Missing: {default_csv}{Style.RESET_ALL}")
                continue
            try:
                out = generate_validation_report(default_csv, report_path)
            except Exception as e:
                print(f"{Fore.RED}{e}{Style.RESET_ALL}")
                continue
            print(f"Report: {out}")

        elif choice == "4":
            print("Exit.")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    try:
        _menu()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
