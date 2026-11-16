# 🧠 Text Topic Modelling Application

An interactive Streamlit app that automatically discovers hidden topics and common themes in a collection of text documents using **LDA** (Latent Dirichlet Allocation) and **NMF** (Non-negative Matrix Factorization).

Upload your own documents or feedback logs, or load the **20 Newsgroups** benchmark dataset. Then clean the text, train a model, explore the topics, and see which topic every document belongs to.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-LDA%20%7C%20NMF-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📑 Table of Contents

- [Features](#-features)
- [Demo / Screenshots](#-demo--screenshots)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Running the App](#-running-the-app)
- [How to Use the GUI](#-how-to-use-the-gui)
- [How It Works](#-how-it-works)
- [Evaluation Metrics](#-evaluation-metrics)
- [Sample Data](#-sample-data)
- [Running Tests](#-running-tests)
- [Limitations and Future Work](#-limitations-and-future-work)
- [Viva / Interview Q&A](#-viva--interview-qa)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

| Area | What you get |
|---|---|
| **Data ingestion** | Multiple `.txt` uploads, CSV / Excel with column picker, 20 Newsgroups (adjustable size and categories), and built-in sample data as an offline fallback |
| **Input safety** | Encoding fallback (UTF-8, then Latin-1), empty-file handling, and a minimum-document check with friendly error messages |
| **Preprocessing** | Lowercasing, URL / email / HTML removal, number and punctuation stripping, tokenization, stopword removal (NLTK plus custom list), lemmatization, short-token filtering. Every step is toggleable from the GUI |
| **Topic models** | LDA on word counts and NMF on TF-IDF, with configurable topics, iterations, `max_df`, `min_df`, n-grams, and random seed |
| **Topic insight** | Top words per topic with normalized weights, plus auto-generated labels such as `space_mission_orbit` |
| **Document grouping** | Dominant topic and confidence score for every document, with search, filter, and CSV export |
| **Evaluation** | Perplexity, log-likelihood, coherence (c_v, via gensim when available), and a "Find Best Number of Topics" tuner |
| **Visualizations** | Top-word bar charts, word clouds, topic distribution (pie / bar), document-topic heatmap, 2D PCA / t-SNE scatter, optional pyLDAvis |
| **Inference** | Paste new text and see its predicted topic distribution using the trained model |

---

## 🖼 Demo / Screenshots

> Add your screenshots to a `docs/` or `assets/` folder and update the paths below.

| Discovered Topics | Document Assignments |
|---|---|
| ![Topics](assets/topics.png) | ![Assignments](assets/assignments.png) |

| Visualizations | Try Your Own Text |
|---|---|
| ![Viz](assets/visualizations.png) | ![Predict](assets/predict.png) |

---

## 🛠 Tech Stack

- **Language:** Python 3.11
- **GUI:** Streamlit
- **NLP / ML:** scikit-learn (`CountVectorizer`, `TfidfVectorizer`, `LatentDirichletAllocation`, `NMF`), NLTK
- **Data:** pandas, numpy, openpyxl
- **Visualization:** Plotly, Matplotlib, WordCloud
- **Optional:** gensim (coherence score), pyLDAvis (interactive topic map)

> **Graceful fallbacks:** if `gensim` or `pyLDAvis` cannot be installed or imported in your environment, the app logs a notice and continues using perplexity / log-likelihood and the native Plotly PCA / t-SNE projection.

---

## 📁 Project Structure

```
topic_modelling_app/
├── app.py                     # Streamlit GUI entry point (6 tabs)
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # TXT, CSV/XLSX, 20 Newsgroups, sample-data loaders + validation
│   ├── preprocessing.py       # Configurable cleaning pipeline (PreprocessingOptions dataclass)
│   ├── topic_model.py         # LDA & NMF training, top words, doc-topic matrix, inference
│   ├── evaluation.py          # Perplexity, log-likelihood, coherence, topic-count search
│   └── visualization.py       # Plotly charts, word clouds, heatmap, PCA / t-SNE
├── sample_data/
│   ├── feedback.csv           # 30 multi-category customer reviews
│   └── txt_docs/              # 12 sample text files (tech, science, health, space, ...)
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   └── test_topic_model.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/topic_modelling_app.git
cd topic_modelling_app
```

### 2. Create a virtual environment (Python 3.11 recommended)

**Windows**
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. NLTK data

The app downloads the required NLTK resources (`stopwords`, `wordnet`, `punkt`, `punkt_tab`) automatically on first run. If you are offline, it falls back to a regex tokenizer and continues to work. To download manually:

```bash
python -m nltk.downloader stopwords wordnet punkt punkt_tab
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 🖱 How to Use the GUI

### Sidebar

1. **Choose a data source:** Upload TXT files, Upload CSV/Excel, 20 Newsgroups, or Load Sample Data.
2. **Set ingestion options:** subset size and categories for 20 Newsgroups, or the text column for CSV / Excel.
3. **Configure preprocessing:** toggle each cleaning step and add custom stopwords.
4. **Configure the model:** algorithm (LDA / NMF), number of topics, top words, iterations, `max_df`, `min_df`, n-gram range, random seed.
5. Click **🚀 Run Topic Modelling**. Use **🔄 Reset** to clear everything.

### Tabs

| Tab | Description |
|---|---|
| **Data Overview** | Total and cleaned document counts, vocabulary size, average length, raw document table, and a before / after cleaning inspector |
| **Discovered Topics** | A card per topic with its label, top words with weights, bar chart, and word cloud |
| **Document Assignments** | Searchable table (Doc ID, Snippet, Dominant Topic, Topic Label, Confidence), a topic filter, and CSV download |
| **Visualizations** | Topic distribution, document-topic heatmap, and 2D PCA / t-SNE scatter colored by topic |
| **Model Evaluation** | Perplexity, log-likelihood, coherence, and the "Find Best Number of Topics" tuning curve with a recommended *k* |
| **Try Your Own Text** | Paste any text to see the cleaned tokens and the predicted topic distribution |

### Things to try

1. Click **Load Sample Data**, set topics to 4, and run. Check whether the topics match the sample categories.
2. Load **20 Newsgroups** with 2,000 documents and compare **LDA vs NMF** results.
3. Open **Model Evaluation** and run **Find Best Number of Topics**, then retrain with the recommended value.

---

## 🔬 How It Works

```
Raw documents
     │
     ▼
Preprocessing ── lowercase → remove URLs/emails/HTML/numbers/punctuation
     │            → tokenize → remove stopwords → lemmatize → drop short tokens
     ▼
Vectorization ── CountVectorizer (LDA)  |  TfidfVectorizer (NMF)
     │
     ▼
Topic model ──── LDA  or  NMF
     │
     ├── Topic-word matrix  → top words + auto labels
     └── Document-topic matrix → dominant topic + confidence
     ▼
Visualization, evaluation, and inference on new text
```

### LDA in simple words

LDA assumes that:

- every **document** is a mixture of a few **topics** (for example, 70% sports, 30% politics), and
- every **topic** is a distribution over **words** (the sports topic favors "game", "team", "season").

Starting from random guesses, LDA repeatedly adjusts these two sets of distributions until they best explain the words actually seen in the documents. The output is:

- a **topic-word** matrix, which tells you what each topic is about, and
- a **document-topic** matrix, which tells you what each document is about.

Two Dirichlet priors control the behavior: **α** (document-topic sparsity) and **β / η** (topic-word sparsity).

### LDA vs NMF in this app

| | LDA | NMF |
|---|---|---|
| Input matrix | Word counts | TF-IDF |
| Approach | Probabilistic generative model | Matrix factorization (`V ≈ W·H`, all values ≥ 0) |
| Output | True probabilities | Non-negative weights (normalized for display) |
| Tends to be better at | Interpretable mixtures on longer texts | Crisp, coherent topics on short texts, and faster |

---

## 📊 Evaluation Metrics

- **Perplexity** (LDA): lower is better. Measures how well the model predicts held-out words.
- **Log-likelihood** (LDA): higher is better.
- **Reconstruction error** (NMF): lower is better.
- **Coherence (c_v)**: higher is better. Measures how semantically related the top words of each topic are (requires `gensim`).
- **Best-k search:** trains models across a range of topic counts (default 2 to 15), plots the metric curves, and recommends a value of *k*.

> Perplexity and human interpretability do not always agree. Coherence is usually the better guide for choosing *k*, and you should still read the topics yourself.

---

## 🗂 Sample Data

Both samples work offline and are used automatically if 20 Newsgroups cannot be downloaded.

- `sample_data/feedback.csv`: 30 customer reviews covering bugs, billing, feature requests, and praise.
- `sample_data/txt_docs/`: 12 text files covering AI, astronomy, renewable energy, medicine, economics, sports, and more.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

| File | Covers |
|---|---|
| `test_preprocessing.py` | Cleaning steps, stopword filtering, lemmatization, edge cases (empty strings, non-ASCII) |
| `test_topic_model.py` | Vectorization, LDA / NMF fitting, top-word extraction, probability normalization, single-text inference |

Quick import smoke test:

```bash
python -c "import app, src.data_loader, src.preprocessing, src.topic_model, src.evaluation, src.visualization; print('Imports valid!')"
```

---

## ⚠️ Limitations and Future Work

**Current limitations**

- LDA and NMF use a bag-of-words view and ignore word order and deeper context.
- Results depend on preprocessing choices and the chosen number of topics.
- Very short documents (tweets, one-line reviews) give weaker topics.
- Very large corpora (10,000+ documents) can make training and t-SNE slow.
- The topic count must be chosen up front.

**Planned improvements**

- [ ] BERTopic / transformer-embedding based topics
- [ ] Dynamic topic modelling (topics over time)
- [ ] Hierarchical topic modelling
- [ ] Multi-language support
- [ ] Save / load trained models
- [ ] Manual topic renaming and merging in the GUI
- [ ] Docker image and cloud deployment

---

## 🎓 Viva / Interview Q&A

<details>
<summary><b>1. What is LDA?</b></summary>

Latent Dirichlet Allocation is an unsupervised, probabilistic topic model. It assumes each document is a mixture of topics and each topic is a distribution over words. It infers both distributions from the corpus alone, with no labels needed.
</details>

<details>
<summary><b>2. What do alpha and beta mean?</b></summary>

They are the parameters of the Dirichlet priors. **Alpha** (`doc_topic_prior`) controls how many topics a document contains: a low alpha means documents focus on few topics, a high alpha means they spread across many. **Beta / eta** (`topic_word_prior`) controls how many words define a topic: a low beta gives topics with few dominant words, a high beta gives broader topics.
</details>

<details>
<summary><b>3. Why remove stopwords?</b></summary>

Words like "the", "is", and "and" appear in nearly every document and carry little meaning. Left in, they would dominate every topic and hide the informative words. Removing them also shrinks the vocabulary and speeds up training.
</details>

<details>
<summary><b>4. Why lemmatize?</b></summary>

Lemmatization maps word forms (running, ran, runs) to a base form (run). This merges counts that would otherwise be split, giving a smaller vocabulary and more stable topics.
</details>

<details>
<summary><b>5. How do you choose the number of topics?</b></summary>

Train models over a range of *k*, compare coherence (and perplexity), pick the point where coherence peaks or levels off, and then check by reading the top words. This app automates the search under **Model Evaluation**.
</details>

<details>
<summary><b>6. What is the difference between LDA and NMF?</b></summary>

LDA is a probabilistic generative model using word counts and Dirichlet priors. NMF is a linear-algebra factorization of a (usually TF-IDF) matrix into two non-negative matrices. NMF is often faster and gives crisper topics on short texts, while LDA gives proper probability distributions.
</details>

<details>
<summary><b>7. What is coherence?</b></summary>

A score of how semantically related the top words in a topic are, usually based on how often they co-occur in the corpus. High coherence means a topic is more human-interpretable. c_v is a common variant.
</details>

<details>
<summary><b>8. Why not rely on perplexity alone?</b></summary>

Lower perplexity means a better statistical fit, but research shows it can correlate poorly with human judgment, and it often keeps improving as *k* grows. Coherence is a better guide for interpretability.
</details>

<details>
<summary><b>9. Why use CountVectorizer for LDA and TF-IDF for NMF?</b></summary>

LDA models word counts as draws from multinomial distributions, so it needs integer counts. NMF works on any non-negative matrix, and TF-IDF down-weights very common words, which usually improves topic quality.
</details>

<details>
<summary><b>10. How does the app predict the topic of new text?</b></summary>

The new text goes through the same preprocessing and the already-fitted vectorizer, and the trained model's `transform` method returns a topic distribution. The highest-probability topic is shown as the dominant topic.
</details>

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

- [scikit-learn](https://scikit-learn.org/) for LDA, NMF, and the 20 Newsgroups dataset
- [NLTK](https://www.nltk.org/) for text preprocessing tools
- [Streamlit](https://streamlit.io/) for the web interface
- [gensim](https://radimrehurek.com/gensim/) and [pyLDAvis](https://github.com/bmabey/pyLDAvis) for coherence and topic visualization

---

⭐ If you found this project useful, consider giving it a star!

