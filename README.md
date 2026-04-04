# Student engagement monitoring — dissertation prototype

Synthetic-data demonstration of a student engagement monitoring concept. **All datasets and names are simulated** (no real students or institutional records).

## Layout

| Path | Purpose |
|------|---------|
| **`data/`** | CSV datasets (`students_data.csv`, `assessments_data.csv`, `interventions_data.csv`) |
| **`charts/`** | Exported PNG visualisations |
| **`data-generation/`** | Generator script, instructions, and `dataset_summary.md` |
| **`risk-scoring-engine/`** | Reserved for risk-scoring implementation (Prompt 1) |
| **`dashboard/`** | Reserved for dashboard UI (Prompt 2) |
| **`documentation/`** | Research form, SRS, and related documents |

## Regenerating data (optional)

From `data-generation/`:

```bash
python generate_student_engagement_dataset.py
```

This overwrites CSVs in `../data/`, charts in `../charts/`, and `dataset_summary.md` in this folder. Use seed **42** in the script for reproducibility.

## Dependencies

Python 3.10+ with `pandas`, `numpy`, `matplotlib`, and `faker` (see `data-generation/DATA_GENERATION_INSTRUCTIONS.md`).
