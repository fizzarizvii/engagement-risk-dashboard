# Data generation instructions — simulated student engagement prototype

This project supports a **dissertation demonstration** of a predictive student engagement monitoring system. **All data are entirely simulated.** No real University of Chester (or any other) student records are used. Names, emails, and IDs are produced with the Faker library and procedural rules.

The generator script lives in **`data-generation/`**. Outputs are written relative to the project root:

| Output | Location |
|--------|----------|
| CSV datasets | **`../data/`** |
| PNG charts | **`../charts/`** |
| Auto summary | **`dataset_summary.md`** (this folder) |

---

## What the pipeline does

Running **`generate_student_engagement_dataset.py`** from **`data-generation/`** (single consolidated script) will:

1. Create **`../data/students_data.csv`** — 500 synthetic students with correlated attendance, VLE use, assessment behaviour, demographics, risk scores, and optional alerts.
2. Create **`../data/assessments_data.csv`** — 30–40 upcoming assessments (due dates spread across about eight weeks from the fixed “as of” date in the script).
3. Create **`../data/interventions_data.csv`** — 200–300 historical intervention rows linked to `student_id` values from the student file.
4. Create **`dataset_summary.md`** (in `data-generation/`) — submission-ready tables and headline statistics (risk breakdown, averages by risk level, recent alerts, intervention outcomes, channel response rates). Markdown opens cleanly in Word or VS Code and can be exported to PDF; it is more readable for examiners than raw JSON.
5. Export **five PNG charts** under **`../charts/`** (matplotlib): risk histogram; attendance vs grade; VLE vs grade; intervention response rates by channel; engagement trend pie chart.

**Reproducibility:** random seed **42** is set for Python `random`, NumPy, and Faker.

**As-of date for summaries:** `2025-03-22` (used for “alerts in the last week” and assessment due-date windows). Change this in the script if you need a different demo timeline.

---

## File: `students_data.csv`

| Column | Description |
|--------|-------------|
| `student_id` | `STU0001` … `STU0500` |
| `student_name` | Faker-generated name (UK locale) |
| `email` | `firstname.lastname@student.chester.ac.uk` (sanitised local part) |
| `program` | One of: Computer Science, Data Science, Business Analytics, Engineering, Psychology |
| `year_of_study` | 1, 2, or 3 |
| `enrollment_date` | Random date in September or October 2024 |
| `attendance_rate` | 0–100%; centred near 75% with spread; **commuters** tend slightly lower |
| `vle_logins_last_week` | 0–25; correlated with attendance (noisy) |
| `vle_time_hours_last_week` | 0–20 hours; correlated with logins; **part-time workers** tend slightly lower |
| `assignments_submitted_on_time` | 0–8 |
| `assignments_submitted_late` | 0–3; on-time + late capped so totals stay realistic |
| `last_assessment_grade` | 0–100; correlated with engagement proxies |
| `previous_semester_gpa` | 0.0–4.0; correlated with current grade |
| `is_commuter` | ~30% true |
| `works_part_time` | ~40% true |
| `has_declared_disability` | ~15% true |
| `is_international` | ~25% true |
| `financial_support` | Full / Partial / None |
| `engagement_trend` | Improving / Stable / Declining — from a **synthetic** latent-based proxy (not real longitudinal logs) |
| `accessed_upcoming_assessment_brief` | Simulated whether the student opened the brief for a nearest upcoming assessment (within ~3 weeks) |
| `days_to_nearest_assessment` | Days until that synthetic nearest deadline (3–21) |
| `risk_score_pre_intervention` | 0–100 composite from engagement metrics only (see **Risk model v2**) |
| `prior_intervention_count` | Count of historical interventions for this student (from `interventions_data.csv`) |
| `prior_interventions_improved` | How many of those showed improved engagement |
| `risk_score` | Final 0–100 score after blending pre-intervention composite with **intervention history stress** |
| `risk_level` | Low (0–33), Medium (34–66), High (67–100) |
| `intervention_history_stress` | 0–100 stress from repeated contact and low improvement rate |
| `last_alert_date` | ISO date for ~30% of students; blank otherwise |
| `alert_response_status` | Responded / No Response / Meeting Scheduled if alerted; blank otherwise |

### Designed cohorts (for realistic demos)

- **~20% at-risk cluster:** lower latent engagement → tends to push attendance, VLE use, and grades down (not perfectly).
- **~5% “resilient” students:** attendance forced into **60–70%** but grades biased **upward** so you can show “doing fine despite weaker attendance” cases.
- Correlations are **intentionally noisy** so the data does not look artificially perfect.
- Interventions are **biased** toward students with higher pre-intervention risk so that outreach history aligns with at-risk profiles.

### Risk model v2 (composite)

The canonical specification lives in **`../risk-scoring-engine/risk_scoring_engine.py`** (model block at the top of the file, shared with the CLI). It combines weighted **component stresses** (attendance, VLE logins, VLE hours, grade, submission behaviour, trajectory, GPA gap, assessment preparation), applies **contextual easing** (commuter, part-time, disability), **calibrates** to 0–100, then blends with **intervention history stress** from prior outreach counts and outcomes. This replaces the older four-threshold additive rule set.

---

## File: `assessments_data.csv`

Demonstrates an “alert before deadline” style view: assessments with enrolment counts and how many students have “accessed the brief” (simulated).

| Column | Description |
|--------|-------------|
| `assessment_id` | `ASS0001`, … |
| `module_code` | e.g. `CS101`, `DA202` |
| `assessment_title` | Drawn from a small realistic pool |
| `due_date` | Within ~8 weeks of the script’s as-of date |
| `assessment_type` | Essay, Project, Exam, Presentation |
| `weight_percentage` | 10, 15, 20, 30, 40, or 50 |
| `students_enrolled` | Simulated cohort size |
| `students_accessed_brief` | Binomial draw from enrolled |
| `students_not_accessed` | Enrolled minus accessed |

---

## File: `interventions_data.csv`

Past outreach records for outcome tracking demos.

| Column | Description |
|--------|-------------|
| `intervention_id` | `INT0001`, … |
| `student_id` | Matches `students_data.csv` |
| `intervention_date` | Within the last ~90 days before as-of date |
| `intervention_type` | Email, SMS, Phone Call, In-Person Meeting |
| `reason` | Low Attendance, Missed Deadline, Declining Grades, No VLE Activity |
| `student_responded` | True/False |
| `response_time_hours` | 0–168 if responded; blank if not |
| `engagement_improved` | **~70%** true if responded; **~20%** true if not |
| `attendance_before`, `attendance_after` | Anchored to the student’s simulated attendance; “after” tends higher if improved |
| `grade_before`, `grade_after` | Anchored to last assessment grade; “after” tends higher if improved |

---

## Output charts (PNG)

| File | Content |
|------|---------|
| `risk_score_distribution.png` | Histogram of `risk_score` |
| `attendance_vs_grade_scatter.png` | Scatter; colour = risk level |
| `vle_vs_grade_scatter.png` | VLE logins vs grade; colour = risk score |
| `intervention_response_rates.png` | Bar chart: response % for Email, SMS, Phone Call |
| `engagement_trend_pie.png` | Pie: Improving / Stable / Declining |

---

## How to re-run

From this folder:

```bash
python generate_student_engagement_dataset.py
```

Dependencies: **Python 3.10+** recommended, plus `pandas`, `numpy`, `matplotlib`, `faker`.

Example install:

```bash
pip install pandas numpy matplotlib faker
```

---

## Ethics and limitations (for your write-up)

- This dataset is **for prototype and methodology illustration only**; it must not be presented as empirical findings about real students.
- **Risk scores are rule-based** and simplified; a production system would use validated models, governance, and human review.
- **Engagement trend** is a **proxy** from cross-sectional fields, not true time-series VLE logs.

---

## Example use cases in a dissertation

- Show how dashboards combine **attendance**, **VLE**, and **assessment** signals.
- Illustrate **threshold-based risk flags** and how **intervention history** could be evaluated.
- Demonstrate **export pipelines** (CSV → summary → figures) for reporting or validation chapters.

If you change `AS_OF_DATE`, `N_STUDENTS`, or cohort proportions, update this document accordingly so it stays aligned with the script.
