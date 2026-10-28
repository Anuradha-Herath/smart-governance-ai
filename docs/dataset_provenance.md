# Dataset Register & Provenance Document

**Research Project**: An AI-Driven Framework for Public, Expert Opinion Mining and Evidence-Based Decision Support in Smart Governance  
**Date**: October 2026  
**Target Case Study**: Sri Lanka Electricity Tariff & Energy Revisions (`DEC_2023_ELEC_01`)

---

## 1. Overview of Data Architecture

The project manages 19 distinct datasets (~1.25 GB, >263,000 documents) stored locally in `Datasets/` and accessed through [`src/utils/data_loader.py`](file:///I:/A_Research%20Project/Project%20folder/src/utils/data_loader.py).

### Data Classification & Processing Status
1. **Ready-to-Use Public Benchmarks**: IBM Debater ArgKP, UKP ASPECT, Sinhala Sentiment (TSV/Parquet), Singlish Sentiment, UOM Code-Mixed corpus.
2. **Pre-Extracted Document Corpora**: Sri Lanka Cabinet Decisions, Extraordinary Gazettes (2010s & 2020s), Acts, Bills, Hansards, CBSL Annual Reports (stored as `docs_all.tsv`, `docs.json`, and `chunks.json`).
3. **Official Indicator Series**: 30 annual indicator time-series (1980–2025) across Economic, Social, Sectoral, and Governance categories.
4. **Targeted Decision Data (To Curate in Phase 3/4)**: Small decision-specific comment sample (300–500 rows) for stance annotation (`Support`/`Oppose`/`Neutral`), and human-verified argument extraction pairs.

---

## 2. Pipeline 1: Multilingual Public Opinion & Stance Mining

### 2.1 Sinhala News Comment Sentiment Dataset
* **Source**: Hugging Face (`Azri-Muhsin/sinhala-news-comment-sentiment` & `sinhala-nlp/sinhala-sentiment-analysis`)
* **Provider**: University of Moratuwa / Sinhala NLP Community
* **Size**: 9,048 comments with train/test splits
* **Languages**: Sinhala (`si`)
* **Format**: Parquet & TSV (`train.tsv`, `test.tsv`)
* **Fields**: `comment` (text), `label` (`positive`, `negative`, `neutral`), article title, date
* **License**: Public Research Dataset (Open Access)
* **Role**: Baseline training and evaluation for Sinhala sentiment classification.
* **Limitations**: General political/economic news comments; labels reflect sentiment, not policy stance toward a specific legislative decision.

### 2.2 Sinhala–English Singlish YouTube Sentiment Dataset
* **Source**: GitHub (`1516CS/Sentiment-dataset-Singlish`)
* **Provider**: 1516CS Research Group
* **Size**: 7,832 manually annotated YouTube comments
* **Languages**: Romanized Sinhala (Singlish) & English
* **Format**: CSV & XLSX
* **Fields**: `comment`, `sentiment` (`Positive`, `Negative`, `Neutral`)
* **License**: Public Research Dataset
* **Role**: Baseline model evaluation for informal code-mixed social media public opinion.
* **Limitations**: Informal orthography, phonetic spelling inconsistencies, sarcasm.

### 2.3 NLPC-UOM Sinhala–English Code-Mixed & Code-Switched Dataset
* **Source**: Hugging Face (`NLPC-UOM/Sinhala-English-Code-Mixed-Code-Switched-Dataset`)
* **Provider**: Natural Language Processing Centre, University of Moratuwa
* **Size**: 10,000 annotated social media comments
* **Languages**: Sinhala, English, and hybrid code-switched text
* **Format**: CSV (sentence-level and word-level annotations)
* **Fields**: `sentence`, `sentiment`, `humour`, `hate_speech`, `language_id`
* **License**: Creative Commons / Research License
* **Role**: Multi-task multilingual NLP experiments, language identification, and sentiment benchmarking.

### 2.4 IBM Debater ArgKP & Claim Stance Dataset
* **Source**: Hugging Face (`NLP-Debater-Project/IBM-Debater-ArgKP`)
* **Provider**: IBM Research
* **Size**: ~27,500 argument–key point pairs
* **Languages**: English (`en`)
* **Format**: CSV (`ArgKP_balanced.csv`, `ArgKP_stance_balanced.csv`)
* **Fields**: `arg_id`, `key_point_id`, `argument`, `key_point`, `stance` (-1 / +1), `label`
* **License**: Research License (IBM Project Debater)
* **Role**: Gold-standard reference for argument-to-key-point matching and stance detection.

---

## 3. Pipeline 2: Expert Opinion & Grounded Argument Mining

### 3.1 Sri Lanka Parliamentary Hansard Corpus (2017–2026)
* **Source**: Hugging Face (`sl-parliamentary-nlp/sl-parliamentary-hansard-17-26`) & Parliament of Sri Lanka
* **Size**: Multi-year speech transcripts (38.2 MB Parquet splits)
* **Languages**: Sinhala, Tamil, English, and code-mixed parliamentary debate
* **Fields**: `speaker`, `party`, `date`, `topic`, `text`, `language`
* **License**: Public Official Record (arXiv:2510.04124)
* **Role**: Primary source for parliamentary policy debates, ministerial justifications, and opposition counterarguments.

### 3.2 Central Bank of Sri Lanka (CBSL) Annual Reports (1950–2023)
* **Source**: CBSL Publications / `nuuuwan/cbsl`
* **Size**: 1,137 reports, 50,802 pre-extracted text chunks
* **Languages**: English (`en`)
* **Format**: TSV metadata, JSON chunks, original PDF links
* **Role**: Authoritative expert economic evaluations, financial analyses of state-owned enterprises (CEB, CPC), and macroeconomic assessments.

### 3.3 Treasury & Presidential Media Division Releases
* **Source**: Ministry of Finance (`lk_treasury`) & PMD (`lk_pmd`)
* **Size**: 179 Treasury documents (212 chunks), 2,182 PMD releases (3,382 chunks)
* **Languages**: Sinhala, Tamil, English
* **Role**: Official fiscal policy announcements, tariff justifications, and cabinet press briefings.

### 3.4 UKP ASPECT Argument Similarity Corpus
* **Source**: TU Darmstadt UKP Lab (`UKPLab/UKP_ASPECT`)
* **Size**: 3,595 sentence pairs across 28 controversial topics
* **Format**: TSV
* **License**: CC BY-NC 3.0
* **Role**: Evaluation benchmark for argument clustering, semantic similarity, and deduplication.

---

## 4. Pipeline 3: Historical Decisions & Impact Analysis

### 4.1 Sri Lanka Cabinet Decisions (2010–2026)
* **Source**: Cabinet Office of Sri Lanka / `nuuuwan/lk_cabinet_decisions`
* **Size**: 11,175 decision summaries, 10,518 text chunks
* **Languages**: Sinhala and English
* **Format**: TSV, JSON (`docs.json`, `chunks.json`)
* **Fields**: `doc_id`, `date_str`, `description`, `decision_details_title`, `decision_details_body`
* **License**: MIT (citation: arXiv:2510.04124)
* **Role**: Official catalogue of historical government decisions, policy instruments, and implementation timelines.

### 4.2 Sri Lanka Extraordinary Gazettes (2010–2026)
* **Source**: Department of Government Printing / `nuuuwan/lk_legal_docs`
* **Size**: 105,892 gazette records, 172,497 text chunks (2010–2019 & 2020–2026 collections)
* **Languages**: Sinhala, Tamil, English
* **Role**: Regulatory change detection, legal instruments, tariff orders, and statutory amendments.

### 4.3 Sri Lanka Acts & Bills
* **Source**: Parliament / Department of Government Printing / `nuuuwan/lk_legal_docs`
* **Size**: 3,979 Acts (1981–2026), 4,236 Bills (2010–2026), 83,778 text chunks
* **Languages**: Sinhala, Tamil, English
* **Role**: Identifying statutory parent laws (e.g., Ceylon Electricity Board Act No. 17 of 1969, Sri Lanka Electricity Act No. 20 of 2009).

### 4.4 Sri Lanka Outcome Indicators (WDI & WGI Time-Series, 1980–2025)
* **Source**: World Bank Open Data, Central Bank of Sri Lanka, Department of Census & Statistics
* **Size**: 835 annual indicator observations across 30 series
* **Categories**:
  * **Economic**: GDP growth, CPI, inflation, tax revenue (% GDP), debt (% GDP), trade balance
  * **Social**: Poverty headcount, Gini coefficient, household final consumption expenditure
  * **Sectoral**: Electricity consumption per capita, electricity access (%), fuel imports (% imports)
  * **Governance (WGI)**: Regulatory Quality, Government Effectiveness, Rule of Law, Corruption Control
* **License**: CC-BY 4.0 Open Data
* **Role**: Pre/post policy outcome indicator analysis over 2-year and 4-year implementation windows.

---

## 5. Research Ethics & Sampling Disclaimers
1. **Sampling Validity**: Online comments and social media datasets reflect self-selected participants with internet access and cannot be generalized to the entire Sri Lankan population.
2. **Causal Attribution**: Indicator trajectories observed before and after a policy decision represent descriptive trends and must not be reported as unconfounded causal effects. Macroeconomic shocks (such as the 2022 debt crisis and COVID-19 pandemic) are noted as explicit confounders.
3. **Traceability**: All extracted claims and arguments must preserve document title, date, URL, and page or chunk number.
