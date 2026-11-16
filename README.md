# 🔬 Text Topic Modelling Studio

An interactive, production-ready Natural Language Processing (NLP) application built with **Streamlit**, **scikit-learn**, and **NLTK** that discovers latent topics, keyword importance distributions, and document thematic clusterings across arbitrary text corpora.

---

## 📌 1. Project Overview

Unstructured text collections (customer reviews, survey responses, news articles, academic publications, support tickets) often contain hidden patterns and recurring themes that are difficult to analyze manually. 

**Text Topic Modelling Studio** automates this discovery process:
1. Ingests raw text from multiple `.txt` files, CSV/Excel spreadsheets, or standard datasets (20 Newsgroups).
2. Cleans and normalizes text through an interactive, toggleable NLTK preprocessing pipeline.
3. Fits **Latent Dirichlet Allocation (LDA)** or **Non-Negative Matrix Factorization (NMF)** models.
4. Computes topic-word distributions, extracts normalized keyword weights, and auto-generates human-readable labels.
5. Assigns every document to its dominant topic with an associated confidence probability score.
6. Presents comprehensive diagnostic charts (word clouds, distribution bars, heatmaps, 2D PCA/t-SNE cluster maps) and an out-of-sample real-time inference playground.

---

## ✨ 2. Key Features

- **Multi-Source Ingestion**:
  - Upload multiple `.txt` files (one document per file) with automatic encoding detection (`UTF-8` and `Latin-1`).
  - Upload CSV or XLSX spreadsheets and select which column contains the document text.
  - Fetch subsets of the academic **20 Newsgroups** benchmark dataset by category.
  - Offline fallback to bundled sample customer reviews (`feedback.csv`) and diverse topic articles (`sample_data/txt_docs/`).
- **Configurable NLP Preprocessing**:
  - Lowercasing, URL/email/HTML stripping, number and punctuation removal.
  - English stopword filtering plus a live, user-editable custom stopwords list.
  - Morphological lemmatization via NLTK WordNet.
  - Interactive "Before vs. After" side-by-side inspection for any document.
- **Dual Topic Modelling Engines**:
  - **LDA (Latent Dirichlet Allocation)**: Probabilistic generative modeling via term count frequencies (`CountVectorizer`).
  - **NMF (Non-Negative Matrix Factorization)**: Linear algebraic parts-based decomposition using `TfidfVectorizer`.
- **Dominant Topic Document Assignment**:
  - Searchable, filterable table linking Document IDs, Titles, Text Snippets, Dominant Topics, and Confidence Scores.
  - Instant one-click CSV export of assigned document records.
- **Rich Visualizations (Plotly & WordCloud)**:
  - Horizontal bar charts of top keywords and normalized weights.
  - High-resolution WordClouds per topic.
  - Corpus composition breakdown (interactive Bar and Donut/Pie charts).
  - Document-Topic assignment probability heatmaps.
  - 2D dimensional reduction projection maps (PCA and t-SNE) colored by topic.
- **Model Evaluation & Optimal Topic Count Search**:
  - Model Perplexity and Log-Likelihood for LDA; Reconstruction Error for NMF.
  - Topic Coherence score ($C_v$) with gensim support.
  - Automated grid search across topic counts ($k_{min}$ to $k_{max}$) with metric tuning curves and automated recommendation.
- **Real-Time Inference Sandbox ("Try Your Own Text")**:
  - Paste arbitrary unseen text to test live classification, displaying dominant topic label, confidence percentage, and full topic probability distribution.

---

## 🏗️ 3. Project Structure

```
topic_modelling_app/
├── app.py                     # Streamlit GUI application entry point
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Multi-format document ingestion & encoding validation
│   ├── preprocessing.py       # Configurable NLTK text cleaning pipeline
│   ├── topic_model.py         # LDA & NMF engines, keyword extraction, document assignment
│   ├── evaluation.py          # Perplexity, log-likelihood, coherence & topic-count search
│   └── visualization.py       # Plotly charts, WordClouds, heatmaps, 2D projections
├── sample_data/
│   ├── feedback.csv           # 30 realistic multi-theme customer reviews
│   └── txt_docs/              # 12 diverse sample articles (AI, space, energy, medicine, finance, sports)
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py  # Unit tests for text cleaning & edge cases
│   ├── test_topic_model.py    # Unit tests for LDA/NMF fit & inference
│   └── test_evaluation.py     # Unit tests for metrics & parameter search
├── requirements.txt           # Pinned, tested dependencies
└── README.md                  # Complete documentation & interview guide
```

---

## 🚀 4. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed.

### Step-by-Step Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd "text topic modelling"
   ```

2. **Create a virtual environment:**
   - **Windows:**
     ```powershell
     py -3.11 -m venv .venv
     .\.venv\Scripts\activate
     ```
   - **macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify the installation by running the test suite:**
   ```bash
   pytest tests/ -v
   ```

---

## 🖥️ 5. How to Run the Application

Launch the Streamlit dashboard using:
```bash
streamlit run app.py
```
Your default browser will automatically open to `http://localhost:8501`.

---

## 🧭 6. How to Use the GUI (Tab-by-Tab Walkthrough)

### 1. Sidebar Controls
- **Data Source**: Pick "Use Bundled Sample Data", upload your own `.txt` files, or upload a CSV/Excel file.
- **Text Preprocessing**: Toggle any cleaning stage (e.g. disable lemmatization or add domain keywords into the Custom Stopwords box).
- **Model Settings**: Choose between **LDA** and **NMF**, set the number of topics $k$ (e.g. 4), and adjust the number of top words.
- Click **🚀 Run Modelling** to process documents and train the model. Click **🔄 Reset** to start fresh.

### 2. Main Tabs
- **📊 Data Overview**: Check summary metrics (document count, vocabulary size), inspect raw document snippets, and compare a document before and after cleaning.
- **💡 Discovered Topics**: Inspect the discovered topics. Switch between the "Detailed Topic Inspector" (bar chart + word cloud + weight table) and "All Topics Combined Grid".
- **📄 Document Assignments**: Filter documents by topic, search by keywords, view the model's confidence progress bar, and download the annotated records as a CSV.
- **📈 Visualizations**: Explore corpus breakdown charts, view the document-topic probability heatmap, and explore the interactive 2D PCA or t-SNE scatter plot.
- **🎯 Model Evaluation**: Inspect Perplexity and Log-Likelihood. Click **⚡ Run Topic Count Grid Search** to evaluate $k \in [2, 7]$ and receive an optimal topic recommendation.
- **✍️ Try Your Own Text**: Paste any new text snippet and click **🔮 Classify Topic** to view its predicted topic distribution and confidence score.

---

## 🧠 7. Explanation of LDA in Simple Words

Imagine a journalist who writes articles on a daily basis. When writing an article:
1. The journalist does not write about only one subject. Instead, each article is a **mixture of different topics** (e.g., 60% Space Exploration, 30% Artificial Intelligence, 10% Government Policy).
2. Each topic is, in turn, characterized by a **mixture of words** (e.g., the Space topic frequently uses *"orbit"*, *"rocket"*, *"satellite"*; the AI topic frequently uses *"neural"*, *"model"*, *"algorithm"*).

**Latent Dirichlet Allocation (LDA)** reverses this process through unsupervised statistical inference:
- It only sees the raw finished documents.
- It asks: *"Given the words that appear together across these documents, what latent (hidden) topics most likely produced this collection, and in what proportions does each document contain them?"*
- Mathematically, LDA models the document-topic distribution ($\theta$) and the topic-word distribution ($\phi$) using Dirichlet priors ($\alpha$ and $\beta$), optimizing them via variational inference or Gibbs sampling so words that frequently co-occur are clustered into coherent themes.

---

## 📊 8. Sample Output Description

Running LDA with $k=4$ topics on the bundled sample customer feedback dataset (`sample_data/feedback.csv`) produces distinct themes:
- **Topic 1 (`crash_export_freeze`)**: Software stability and runtime bugs (crashes, memory leaks, freezes).
- **Topic 2 (`charge_invoice_refund`)**: Billing and payment concerns (double charges, subscription renewals, invoice disputes).
- **Topic 3 (`response_ticket_support`)**: Customer service responsiveness (ticket turnaround time, delayed emails, helpdesk articles).
- **Topic 4 (`dashboard_export_shortcut`)**: Feature requests and UI enhancements (dark mode, Excel export, keyboard shortcuts).

In the **Document Assignments** tab, an input review like *"The app crashed when exporting large reports"* receives a **94% confidence score** assigned to Topic 1, allowing automated ticket routing and thematic aggregation.

---

## 🔮 9. Limitations & Future Improvements

1. **Bag-of-Words Limitation**: Standard LDA and NMF treat documents as bags of words, ignoring sequential word order and sentence context.
   - *Future Upgrade*: Integrate transformer-based topic models like **BERTopic** or **Top2Vec**, which use dense sentence embeddings (e.g., MiniLM) and HDBSCAN clustering.
2. **Static Corpus Assumption**: Classical LDA assumes document topics are constant over time.
   - *Future Upgrade*: Implement **Dynamic Topic Modeling (DTM)** to observe how themes evolve over days, months, or years.
3. **Fixed Hyperparameter $k$**: The user must specify the number of topics upfront.
   - *Mitigation*: Our built-in **Topic Count Grid Search** tool automates finding the optimal $k$ across perplexity and coherence curves.
4. **Short Text Sparsity**: For very short texts (tweets, single-sentence chats), word co-occurrence counts are sparse.
   - *Mitigation*: Enable N-gram combinations (unigrams + bigrams) and TF-IDF weighting in the Advanced settings.

---

## 🎓 10. Viva / Technical Interview Q&A

### Q1: What is Latent Dirichlet Allocation (LDA) and how does it work?
> **Answer**: LDA is an unsupervised generative probabilistic model for discovering thematic structures in text collections. It posits that every document is a multinomial distribution over latent topics ($\theta_d \sim \text{Dirichlet}(\alpha)$), and each topic is a multinomial distribution over vocabulary words ($\phi_k \sim \text{Dirichlet}(\beta)$). By observing the word frequencies across documents, LDA uses algorithms such as Variational Expectation-Maximization or Collapsed Gibbs Sampling to invert the generative process and estimate the latent topic mixtures.

### Q2: What do the hyperparameters $\alpha$ (alpha) and $\beta$ (beta / eta) represent?
> **Answer**:
> - **$\alpha$ (Doc-Topic Prior)**: Controls document sparsity over topics. A high $\alpha$ implies documents contain an even mix of many topics; a low $\alpha$ forces documents to be dominated by only one or two specific topics.
> - **$\beta$ (Topic-Word Prior)**: Controls topic sparsity over vocabulary words. A high $\beta$ implies topics contain broad mixtures of most words; a low $\beta$ forces each topic to be characterized by a sharp, concentrated subset of distinct keywords.

### Q3: Why is removing stopwords critical in topic modelling, and what happens if you don't?
> **Answer**: Common function words (e.g., *"the"*, *"is"*, *"and"*, *"with"*) have extremely high corpus document frequencies but carry almost zero topical meaning. If stopwords are not removed, they dominate the count matrix and saturate the top word weights across all topics. This creates massive overlap between topics, collapsing topic separation and rendering the output uninterpretable to human analysts.

### Q4: How do you choose the optimal number of topics ($k$)?
> **Answer**: Choosing $k$ requires balancing mathematical metrics with qualitative interpretability:
> 1. **Topic Coherence ($C_v$ or $U_{mass}$)**: Measures semantic similarity between the high-scoring words of a topic. Higher coherence generally correlates with human interpretability.
> 2. **Perplexity / Hold-out Log-Likelihood**: Measures how well the model predicts held-out documents. Lower perplexity indicates better statistical fit, though very low perplexity can sometimes lead to over-fragmented, uninterpretable micro-topics.
> 3. **The Elbow / Peak Method**: Run a grid search over $k \in [2, 15]$ (as implemented in Tab 5 of this application), plot the metric curves, and choose the $k$ at peak coherence or the elbow of the perplexity curve.

### Q5: What is the key difference between LDA and NMF?
> **Answer**:
> - **Mathematical Framework**: LDA is a **probabilistic** model based on Bayesian inference and Dirichlet distributions; NMF is a **linear algebraic** matrix factorization technique that decomposes a term-document matrix $V \approx W \times H$ under non-negativity constraints ($W \ge 0, H \ge 0$).
> - **Input Representation**: LDA traditionally operates on raw count matrices (`CountVectorizer`) because it models discrete word draws; NMF operates effectively on continuous, weighted matrices such as `TfidfVectorizer`.
> - **Output Interpretation**: LDA output values naturally represent well-calibrated probabilities that sum to 1.0; NMF produces parts-based additive component activations that require normalization for probabilistic interpretation.
> - **Speed**: NMF is typically much faster to converge than LDA on small-to-medium corpora.

### Q6: What is Topic Coherence ($C_v$) and why is it preferred over Perplexity?
> **Answer**: While perplexity evaluates how well a statistical model fits the data distribution, empirical research (e.g., Chang et al., *"Reading Tea Leaves: How Humans Interpret Topic Models"*) proved that models with the lowest perplexity often produce topics that humans judge to be nonsensical. 
> 
> **Topic Coherence ($C_v$)** directly measures the degree of semantic similarity among the top $N$ keywords within a topic using normalized point-wise mutual information (NPMI) and cosine similarity over sliding document windows. High coherence ensures that the top words actually co-occur together in meaningful human contexts.
