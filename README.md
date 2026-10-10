# Layer Health ML/DS - Take-Home Assignment

A machine learning system for extracting structured information (diabetes, smoking, cancer status) from clinical notes using LLMs.

## Repository Structure

```
.
├── data/
│   ├── Pt_001/              # Patient clinical notes (100 patients)
│   │   ├── note_1_*.md
│   │   ├── note_2_*.md
│   │   └── ...
│   └── labels.csv           # Ground truth classifications
│
├── src/
│   ├── models.py            # Pydantic data models
│   ├── utils.py             # File I/O utilities
│   ├── baseline.py          # Baseline prediction function
│   └── predict.py           # Batch prediction runner
│
└── requirements.txt         # Python dependencies
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY='your-key'
cd src
```

## Usage

**Single patient prediction:**
```bash
python baseline.py
```

**Batch predictions:**
```bash
python predict.py
```

## Data Format

### Ground Truth Labels (`data/labels.csv`)
```csv
patient_id,diabetes,smoking,cancer
001,Prediabetes,Not Smoker,Active
002,Type 2 Diabetes,Smoker,No Cancer/Benign
...
```

### Predictions Output

The baseline and labels use zero-padded patient IDs (e.g. `001`); note directories
use a `Pt_` prefix (e.g. `Pt_001`). Retain leading zeros when reading CSVs and
normalize identifiers before joining any custom outputs.

```csv
patient_id,diabetes,smoking,cancer,reasoning,supporting_quotes
001,Prediabetes,Not Smoker,Active,"Clinical reasoning...","Quote 1 | Quote 2"
...
```

## Classification Categories

**Diabetes:** `No Diabetes` | `Prediabetes` | `Type 1 Diabetes` | `Type 2 Diabetes`

**Smoking:** `Smoker` | `Not Smoker`

**Cancer:** `No Cancer/Benign` | `Indeterminate` | `Active` | `Remission`

---

