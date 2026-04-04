# Simulated student engagement dataset — summary

_Generated for demonstration purposes only. As-of date used in summaries: **2025-03-22**._

## Overview

- **Total students:** 500
- **Risk level — Low:** 308
- **Risk level — Medium:** 170
- **Risk level — High:** 22

## Average attendance and grades by risk level

| Risk level | Avg attendance (%) | Avg last assessment grade |
|------------|-------------------:|--------------------------:|
| Low | 82.86 | 75.73 |
| Medium | 68.42 | 53.01 |
| High | 55.18 | 35.05 |

## Alerts

- **Students with an alert dated in the last 7 days (inclusive):** 22

## Interventions (historical sample)

- **Total intervention records:** 248
- **Engagement improved (any intervention):** 45.2%
- **Response rate — Email:** 55.9%
- **Response rate — SMS:** 50.8%
- **Response rate — Phone Call:** 54.4%

## How to interpret risk scores

**Composite model (v2):** weighted blend of eight 0–100 stress components
(attendance, VLE logins, VLE hours, grade, submission behaviour, trajectory,
GPA gap, assessment preparation), with contextual easing for commuters,
part-time workers, and declared disability, then blended with
**intervention history stress** derived from past outreach counts and outcomes.

**Risk level bands:** Low 0–33, Medium 34–66, High 67–100.

---

_This file is produced automatically by `generate_student_engagement_dataset.py`._