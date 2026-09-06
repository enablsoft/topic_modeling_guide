# Topic Modeling Guide

A comprehensive, interactive topic modeling suite built with [Streamlit](https://streamlit.io/). It brings together five major topic modeling algorithms under one roof, with support for multiple embedding backends, LLM-powered topic labeling, advanced text preprocessing, and a cross-model master topic list that deduplicates and merges topics across all models.

**Authors:** Sridhar Nerur and [Claude](https://claude.ai) (Anthropic) — Claude did most of the heavy lifting.

> **Purpose:** This project is intended for **educational and research purposes**. If you use this software or its outputs in academic research, please acknowledge the source (see [Citation](#citation) below).

---

## Features at a Glance

| Capability | Details |
|---|---|
| **5 Topic Models** | LDA, BERTopic, Turftopic (5 variants), BunkaTopics, Structural Topic Modeling (STM) |
| **4 Embedding Backends** | SBERT (local), OpenAI, Voyage AI, Google Gemini |
| **LLM Topic Labeling** | OpenAI-powered topic naming and description for BERTopic, Turftopic, and BunkaTopics |
| **Auto Topic Detection** | BERTopic (HDBSCAN), Turftopic GMM/KeyNMF/Clustering (`n_components="auto"`) |
| **R Integration** | Full STM via `rpy2`, `ldatuning::FindTopicsNumber` with 4 metrics (Griffiths, CaoJuan, Arun, Deveaud) |
| **Text Preprocessing** | Lowercase, punctuation/number removal, stopword removal, lemmatization (spaCy), content-words-only filter |
| **Master Topic List** | Embedding-based deduplication across all models with adjustable similarity threshold |
| **Visualizations** | Word clouds, heatmaps, topic networks, pyLDAvis, BERTopic interactive plots, evolution over time |
| **Export** | Jupyter notebook generation, ZIP download with interactive HTML visualizations, CSVs, and R script for STM |

---

## Architecture

```
topic_modeling/
├── app.py                        # Streamlit application (main entry point)
├── requirements.txt              # Python dependencies
├── README.md
└── utils/
    ├── data_loader.py            # CSV/folder loading, preprocessing, chunking
    ├── embeddings.py             # SBERT, OpenAI, Voyage, Gemini wrappers
    ├── lda_modeling.py           # LDA + Python/R topic optimization
    ├── bertopic_modeling.py      # BERTopic + LLM labeling
    ├── turftopic_modeling.py     # Turftopic (S3, GMM, CTM, KeyNMF, FASTopic) + LLM
    ├── bunka_modeling.py         # BunkaTopics + LLM labeling
    ├── stm_modeling.py           # STM via rpy2 (R) with NMF fallback
    ├── topic_masterlist.py       # Cross-model topic deduplication & merging
    ├── visualization.py          # Shared plotting utilities
    ├── export.py                 # Notebook generation & ZIP export
    └── sample_data.py            # Synthetic dataset generators
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/snerur/topic_modeling_guide.git
cd topic_modeling_guide
```

### 2. Create a virtual environment (recommended)

```bash
conda create -n topicmodel python=3.11 -y
conda activate topicmodel
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install spaCy model (for preprocessing)

```bash
python -m spacy download en_core_web_sm
```

### 5. (Optional) Install R packages for STM and ldatuning

If you have R installed:

```r
install.packages(c("stm", "topicmodels", "slam", "tm"))

# ldatuning may not be on CRAN for newer R versions; install from GitHub:
install.packages("remotes")
remotes::install_github("nikita-moor/ldatuning")
```

Then install the Python-R bridge:

```bash
pip install rpy2
```

### 6. (Optional) API keys for cloud embeddings / LLM labeling

- **OpenAI** — for OpenAI embeddings and/or LLM topic labeling
- **Voyage AI** — for Voyage embeddings
- **Google Gemini** — for Gemini embeddings

API keys are entered in the app's sidebar at runtime; nothing is stored on disk.

---

## Usage

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Workflow

1. **Load data** — Upload a CSV, point to a folder of text files, or use a built-in sample dataset.
2. **Select text columns** — Choose which columns to combine into the corpus.
3. **Preprocessing** (optional) — Enable lowercase, punctuation/number removal, stopword removal, lemmatization, and/or content-words-only filtering.
4. **Choose embedding model** — SBERT (local) or an API-based model.
5. **Select topic models** — Pick one or more: LDA, BERTopic, Turftopic, BunkaTopics, STM.
6. **Configure settings** — Set number of topics, enable auto-detection, toggle LLM labeling, tune model-specific parameters.
7. **Run** — Click "Run Topic Modeling" and explore results across tabs.
8. **Master Topic List** — Review the deduplicated cross-model topic list at the bottom.
9. **Download** — Click "Generate Download Package" to export a ZIP containing interactive HTML visualizations, CSVs, and a Jupyter notebook. Results remain visible while the ZIP is being built.

---

## Topic Models

### LDA (Latent Dirichlet Allocation)
- **Gensim LdaMulticore** — parallel variational Bayes, no external dependencies
- Configurable passes (training epochs) and E-step iterations; automatically uses all available CPU cores minus one
- **Topic number optimization**: Python-based (perplexity, log-likelihood, UMass coherence with elbow detection) and R-based (`ldatuning` with Griffiths 2004, CaoJuan 2009, Arun 2010, Deveaud 2014)
- pyLDAvis interactive visualization (rendered via `pyLDAvis.gensim_models`)
- Word clouds, topic correlation heatmap, document-topic matrix

### BERTopic
- Transformer-based topic modeling with HDBSCAN clustering
- **Auto-detects** number of topics by default
- KeyBERT-inspired representations + optional LLM labeling via OpenAI
- Rich built-in visualizations: barchart, intertopic distance map, hierarchy, heatmap, document scatter, topic tree

### Turftopic
Five model types, each with a different algorithmic approach:
- **Semantic Signal Separation (S3)** — ICA on embeddings
- **Gaussian Mixture Model (GMM)** — soft clustering (supports `auto` topics)
- **Clustering Topic Model (CTM)** — HDBSCAN-based (supports `auto` topics)
- **KeyNMF** — keyword extraction + NMF (supports `auto` topics)
- **FASTopic** — autoencoder-based

LLM labeling via `turftopic.analyzers.OpenAIAnalyzer` generates topic names and descriptions.

### BunkaTopics
- Embedding-based topic modeling with built-in 2D topic map visualization
- LLM labeling via `bunka.get_clean_topic_name()` with langchain's `ChatOpenAI`

### Structural Topic Modeling (STM)
- Full R-based STM via `rpy2` with metadata-aware prevalence modeling
- Falls back to Python NMF when R is not available
- Semantic coherence vs. exclusivity quality plots
- `searchK` for optimal topic number selection
- Generates a standalone R script for reproducibility

---

## LLM Topic Labeling

When enabled, an LLM generates human-readable topic names and descriptions. Three providers are supported, each with a **Verify Connection** button to test the API key before running:

| Provider | Models available |
|---|---|
| **OpenAI** | gpt-4o-mini, gpt-4o, gpt-4.1-nano, gpt-4.1-mini |
| **Anthropic (Claude)** | claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-7 |
| **Google Gemini** | gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash |

How each model is labeled:

| Topic Model | Mechanism |
|---|---|
| BERTopic | `bertopic.representation.OpenAI` — regenerates topic representations |
| Turftopic | `turftopic.analyzers.OpenAIAnalyzer` — returns names, descriptions, and optional document summaries |
| BunkaTopics | `bunka.get_clean_topic_name(ChatOpenAI(...))` — refines keyword-based names |

Results appear in dedicated "LLM Labels" tabs and are incorporated into the master topic list.

---

## Download Package

Click **Generate Download Package** in the app to produce a ZIP containing:

```
topic_modeling_results.zip
├── topic_modeling_analysis.ipynb   # Runnable Jupyter notebook
├── analysis_config.json            # Settings used for the run
├── lda/
│   ├── figures/                    # Interactive HTML charts + PNG word clouds
│   │   ├── topic_words.html
│   │   ├── topic_distribution.html
│   │   ├── topic_correlations.html
│   │   ├── doc_topic_heatmap.html
│   │   ├── topic_network.html
│   │   ├── topic_tuning.html       # (if auto-optimization was run)
│   │   ├── ldatuning_results.html  # (if R ldatuning was run)
│   │   └── wordcloud_topic_N.png
│   ├── document_topic_matrix.csv
│   └── topics_summary.csv
├── bertopic/
│   ├── figures/
│   │   ├── barchart.html
│   │   ├── intertopic_map.html
│   │   ├── hierarchy.html
│   │   ├── heatmap.html
│   │   ├── documents.html
│   │   └── topic_correlations.html
│   ├── document_topic_matrix.csv
│   ├── topics_summary.csv
│   ├── topic_info.csv
│   └── document_info.csv
├── turftopic/figures/ ...
├── bunkatopics/figures/ ...
└── stm/figures/ ...
```

The HTML files are fully self-contained (Plotly loaded from CDN) and interactive — open them directly in any browser.

---

## Master Topic List

After all models run, a **cross-model deduplication** step produces a non-overlapping master list:

1. **Collect** — Gather all topics (with words) from every model.
2. **Embed** — Encode each topic's word string using sentence-transformers.
3. **Similarity** — Compute pairwise cosine similarity across all topics.
4. **Cluster** — Greedy union-find merges topics above the similarity threshold (adjustable, default 0.65).
5. **Merge** — Combine words from clustered topics, deduplicate, and carry over LLM labels.

The result is a single table showing master topics, their constituent words, which models contributed, and any LLM-generated labels. Includes a similarity heatmap and a cluster network graph.

---

## Text Preprocessing

The optional preprocessing pipeline (enabled via checkbox in the sidebar):

| Step | Description |
|---|---|
| Lowercase | Convert all text to lowercase |
| Remove punctuation | Strip all non-alphanumeric characters |
| Remove numbers | Remove standalone numeric tokens |
| Remove stopwords | NLTK English stopwords + optional custom list |
| Lemmatize | spaCy-based lemmatization (e.g., "running" to "run") |
| Content words only | Keep only nouns, verbs, adjectives, and adverbs (spaCy POS tags) |

Custom stopwords can be added as a comma-separated list.

---

## Sample Datasets

Three built-in synthetic datasets for experimentation:

1. **20 Newsgroups (subset)** — ~480 documents across 6 categories
2. **UN General Debate Speeches** — 200 synthetic speeches with country, region, year metadata
3. **Research Abstracts** — 200 abstracts across 5 disciplines with year, journal, citations

---

## Requirements

- Python 3.10+
- R 4.x (optional, for STM and ldatuning)
- See `requirements.txt` for the full list of Python packages

---

## Algorithm Acknowledgments

This software integrates topic modeling algorithms developed by other researchers. We gratefully acknowledge the original authors of these algorithms. If you use this software, please cite the original papers for the underlying methods:

### LDA (Latent Dirichlet Allocation)
Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). **Latent Dirichlet Allocation.** *Journal of Machine Learning Research*, 3, 993–1022. [PDF](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)

### BERTopic
Grootendorst, M. (2022). **BERTopic: Neural topic modeling with a class-based TF-IDF procedure.** *arXiv preprint arXiv:2203.05794*. [PDF](https://arxiv.org/pdf/2203.05794)

### BunkaTopics
de Dampierre, C., Mogoutov, A., and Baumard, N. (2024). **Bunka: Robust and Explainable Large Language Models Cartography.** *arXiv preprint arXiv:2406.06574*. [PDF](https://arxiv.org/pdf/2406.06574)

### Turftopic
Kardos, M., Kostkan, J., Vermillet, A., Nielbo, K., Enevoldsen, K., & Rocca, R. (2025). **Turftopic: Topic Modelling with Contextual Representations from Sentence Transformers.** *Journal of Open Source Software*. [PDF](https://joss.theoj.org/papers/10.21105/joss.08183.pdf)

### Structural Topic Modeling (STM)
Roberts, M. E., Stewart, B. M., Tingley, D., Lucas, C., Leder-Luis, J., Gadarian, S. K., Albertson, B., & Rand, D. G. (2014). **Structural Topic Models for Open-Ended Survey Responses.** *American Journal of Political Science*, 58(4), 1064–1082. [PDF](https://gregoryeady.com/SocialMediaDataCourse/assets/readings/Text_Analysis_II/Roberts,%20Margaret%20E.%20et%20al.%20-%202014%20-%20Structural%20Topic%20Models%20for%20Open-Ended%20Survey%20Responses.pdf)

---

## Citation

If you use this software in academic research, please acknowledge the source:

```
Nerur, S. and Claude (Anthropic). (2025). Topic Modeling Guide: A Comprehensive
Interactive Suite for Topic Modeling. GitHub repository.
https://github.com/snerur/topic_modeling_guide
```

BibTeX:

```bibtex
@software{nerur2025topicmodeling,
  author       = {Nerur, Sridhar and {Claude (Anthropic)}},
  title        = {Topic Modeling Guide: A Comprehensive Interactive Suite for Topic Modeling},
  year         = {2025},
  url          = {https://github.com/snerur/topic_modeling_guide},
  note         = {Educational software for topic modeling research}
}
```

---

## License

This project is provided for educational and research purposes. Please acknowledge the source if used in academic work.
