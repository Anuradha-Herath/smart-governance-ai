# An AI-Driven Framework for Public, Expert Opinion Mining and Evidence-Based Decision Support in Smart Governance

A source-grounded, multi-pipeline research framework designed to assist policymakers by synthesizing public sentiment, expert arguments, and historical government policy outcomes.

---

## 1. System Architecture

```
                    ┌──────────────────────────────┐
                    │   PROPOSED POLICY DECISION   │
                    └──────────────┬───────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
   [ Pipeline 1 ]            [ Pipeline 2 ]            [ Pipeline 3 ]
Multilingual Public       Text-Based Expert         Historical Policy
  Opinion & Stance        Grounded Evidence        Outcome & Indicators
(si / ta / en / Singlish) (Hansard, CBSL, Briefs) (Gazettes, Acts, Macro WDI)
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │      INTEGRATION LAYER       │
                    │   Evidence-Based Synthesis   │
                    │  (Streamlit Decision Brief)  │
                    └──────────────────────────────┘
```

- **Pipeline 1 (Public Opinion & Stance Mining)**: Separates general sentiment from policy stance across Sinhala, Tamil, English, and Singlish. Reports sampling limitations transparently.
- **Pipeline 2 (Expert Opinion & Grounded Extraction)**: Extracts verifiable claims, risks, benefits, and recommendations from official reports and debates with strict page/passage attribution.
- **Pipeline 3 (Historical Decision & Outcome Analysis)**: Retrieves comparable past decisions and compares macroeconomic and social indicators across pre/post windows without making unsubstantiated causal claims.
- **Integration Layer**: Combines all three streams into an interpretable decision-support dashboard.

---

## 2. Directory Structure

```text
smart-governance-ai/
├── configs/              # Configuration files (YAML / JSON)
├── data/
│   ├── raw/              # Immutable raw datasets per pipeline
│   ├── interim/          # Cleaned and normalized intermediate data
│   ├── processed/        # Verified features and model inputs
│   └── external/         # Supporting external references
├── src/                  # Reusable production source code
│   ├── common/           # Shared models, schemas, and base classes
│   ├── pipeline1_public_opinion/
│   ├── pipeline2_expert_opinion/
│   ├── pipeline3_historical_analysis/
│   ├── integration/      # Synthesis and multi-pipeline fusion logic
│   ├── api/              # Optional FastAPI endpoints
│   └── utils/            # Path helpers, logging, text cleaning
├── notebooks/            # Exploratory data analysis & experiment logs
├── annotations/          # Human-verified labels and evaluation gold sets
├── models/               # Saved model checkpoints and tokenizer weights
├── outputs/
│   ├── figures/          # Visualizations and publication charts
│   ├── evaluation/       # Performance metrics (precision, recall, F1)
│   └── reports/          # Generated decision-support reports
├── tests/                # Automated pytest unit and integration tests
├── app/                  # Streamlit dashboard and pages
├── docs/                 # Methodology documentation and user guides
├── .env.example          # Environment variable template
├── .gitignore            # Git exclusion rules
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 3. Quickstart Guide (Windows)

### Prerequisites
- Windows 10/11 64-bit
- Python 3.10 – 3.12
- Git for Windows
- Visual Studio Code

### Step 1: Clone and Enter Directory
```powershell
cd "I:\A_Research Project\Project folder"
```

### Step 2: Activate the Virtual Environment
**PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`)*

**Command Prompt (cmd):**
```cmd
.venv\Scripts\activate.bat
```

### Step 3: Install Core Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation
```powershell
pytest tests/
```

---

## 4. Research Integrity & Safeguards
- **Correlation vs. Causation**: Pipeline 3 documents observed before-and-after trajectories. Observed changes are not declared as causal effects without a formal quasi-experimental design.
- **Representativeness**: Pipeline 1 online comments reflect self-selected social media participants and are explicitly disclaimed as non-representative of the general population.
- **Grounding**: Pipeline 2 extracts arguments strictly tied to source documents and page numbers to prevent hallucination.
