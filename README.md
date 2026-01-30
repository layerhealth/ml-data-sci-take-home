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
│   ├── predict.py           # Batch prediction runner
│   └── requirements.txt     # Python dependencies
│
└── generation/
    ├── labels.csv           # Original detailed labels
    └── generate_clinical_notes.py
```

## Setup

```bash
cd src
pip install -r requirements.txt
export ANTHROPIC_API_KEY='your-key'
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

