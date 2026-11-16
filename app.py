"""Streamlit Application for Text Topic Modelling Studio.

A production-grade, interactive application for discovering latent topics,
visualizing keyword distributions, assigning documents, evaluating models,
and testing real-time inference using Latent Dirichlet Allocation (LDA)
and Non-Negative Matrix Factorization (NMF).
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Add current directory to path to support src imports
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data_loader import (
    AVAILABLE_20NEWS_CATEGORIES,
    load_20newsgroups,
    load_csv_or_excel,
    load_sample_fallback,
    load_uploaded_txts,
    validate_documents,
)
from src.evaluation import (
    calculate_lda_metrics,
    calculate_nmf_metrics,
    compute_coherence_score,
    search_best_topic_count,
)
from src.preprocessing import (
    PreprocessingOptions,
    TextPreprocessor,
    ensure_nltk_resources,
)
from src.topic_model import TopicModelEngine
from src.visualization import (
    generate_pyldavis_html,
    generate_wordcloud_figure,
    plot_all_topics_grid,
    plot_document_distribution,
    plot_document_topic_heatmap,
    plot_documents_2d_projection,
    plot_top_words_bar,
    plot_topic_search_curves,
)


# --- PAGE SETUP ---
st.set_page_config(
    page_title="Text Topic Modelling Studio",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished dashboard look
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .topic-pill {
        background-color: #e3f2fd;
        color: #0d47a1;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9em;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 6px 6px 0 0;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_session_state() -> None:
    """Initialize default keys in Streamlit session state."""
    defaults: Dict[str, Any] = {
        "raw_df": None,
        "processed_df": None,
        "dropped_df": None,
        "preprocessor": None,
        "model_engine": None,
        "assigned_df": None,
        "lda_metrics": None,
        "coherence_info": None,
        "is_model_trained": False,
        "eval_search_results": None,
        "eval_recommendation": None,
        "data_source_name": "None",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


initialize_session_state()


# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.title("⚙️ Control Panel")

    # 1. DATA SOURCE SELECTION
    st.subheader("1. Data Ingestion")
    data_source = st.selectbox(
        "Select Data Source",
        [
            "Use Bundled Sample Data",
            "Upload TXT Files",
            "Upload CSV / Excel",
            "20 Newsgroups Dataset",
        ],
        index=0,
        help="Choose how you want to provide text documents for topic discovery.",
    )

    loaded_df: Optional[pd.DataFrame] = None
    data_source_label = data_source

    if data_source == "Use Bundled Sample Data":
        sample_choice = st.radio(
            "Sample Dataset",
            ["Multi-Topic Articles (12 TXT docs)", "Customer Feedback (30 CSV reviews)"],
            index=1,
        )
        if sample_choice == "Customer Feedback (30 CSV reviews)":
            feedback_path = BASE_DIR / "sample_data" / "feedback.csv"
            if feedback_path.exists():
                try:
                    loaded_df, _ = load_csv_or_excel(feedback_path, "feedback.csv", text_column="text")
                    data_source_label = "Sample Customer Feedback (30 reviews)"
                except Exception as e:
                    st.error(f"Error loading sample feedback: {e}")
        else:
            loaded_df = load_sample_fallback(base_dir=BASE_DIR)
            data_source_label = "Sample Multi-Topic Articles (12 docs)"

    elif data_source == "Upload TXT Files":
        uploaded_files = st.file_uploader(
            "Upload .txt Documents",
            type=["txt"],
            accept_multiple_files=True,
            help="Upload multiple .txt files. Each file represents an individual document.",
        )
        if uploaded_files:
            loaded_df = load_uploaded_txts(uploaded_files)
            data_source_label = f"Uploaded TXT Files ({len(uploaded_files)} files)"

    elif data_source == "Upload CSV / Excel":
        uploaded_table = st.file_uploader(
            "Upload Tabular File",
            type=["csv", "xlsx", "xls"],
            help="Upload a CSV or Excel spreadsheet containing document text.",
        )
        if uploaded_table:
            try:
                # Peek at column names
                raw_preview, all_cols = load_csv_or_excel(
                    uploaded_table, uploaded_table.name
                )
                selected_col = st.selectbox(
                    "Choose Text Column",
                    all_cols,
                    index=0,
                    help="Select the column header containing the text documents.",
                )
                if hasattr(uploaded_table, "seek"):
                    uploaded_table.seek(0)
                loaded_df, _ = load_csv_or_excel(
                    uploaded_table, uploaded_table.name, text_column=selected_col
                )
                data_source_label = f"{uploaded_table.name} (Col: {selected_col})"
            except Exception as e:
                st.error(f"Error reading file: {e}")

    elif data_source == "20 Newsgroups Dataset":
        st.caption("Standard academic text classification and topic modeling benchmark.")
        subset_size = st.slider(
            "Sample Size (documents)",
            min_value=200,
            max_value=3000,
            value=800,
            step=100,
            help="Number of documents to sample from 20 Newsgroups.",
        )
        selected_cats = st.multiselect(
            "Filter Categories (Optional)",
            options=AVAILABLE_20NEWS_CATEGORIES,
            default=[
                "comp.graphics",
                "rec.sport.baseball",
                "sci.space",
                "talk.politics.mideast",
            ],
            help="Select specific categories or leave empty to sample from all 20 categories.",
        )
        if st.button("📥 Fetch 20 Newsgroups Data", use_container_width=True):
            with st.spinner("Fetching 20 Newsgroups dataset (with offline fallback)..."):
                df_20, is_fallback = load_20newsgroups(
                    subset_size=subset_size,
                    categories=selected_cats if selected_cats else None,
                    base_dir=BASE_DIR,
                )
                loaded_df = df_20
                if is_fallback:
                    st.warning("Network unavailable. Loaded local fallback sample documents.")
                else:
                    st.success(f"Fetched {len(loaded_df)} documents successfully!")
                data_source_label = f"20 Newsgroups ({len(loaded_df)} docs)"

    # Automatically store new dataset if changed
    if loaded_df is not None and not loaded_df.empty:
        if (
            st.session_state.raw_df is None
            or len(st.session_state.raw_df) != len(loaded_df)
            or st.session_state.data_source_name != data_source_label
        ):
            st.session_state.raw_df = loaded_df
            st.session_state.data_source_name = data_source_label
            st.session_state.is_model_trained = False
            st.session_state.processed_df = None
            st.session_state.model_engine = None

    # 2. PREPROCESSING OPTIONS
    st.divider()
    st.subheader("2. Text Preprocessing")
    with st.expander("Pipeline Cleaning Toggles", expanded=False):
        opt_lower = st.checkbox("Lowercase text", value=True)
        opt_urls = st.checkbox("Remove URLs", value=True)
        opt_emails = st.checkbox("Remove Email Addresses", value=True)
        opt_html = st.checkbox("Remove HTML Tags", value=True)
        opt_numbers = st.checkbox("Remove Numbers", value=True)
        opt_punct = st.checkbox("Remove Punctuation & Special Chars", value=True)
        opt_stopwords = st.checkbox("Remove English Stopwords", value=True)
        opt_lemmatize = st.checkbox("Lemmatize Tokens (WordNet)", value=True)
        opt_min_len = st.number_input("Min Token Length", min_value=1, max_value=8, value=3)

    custom_stops_raw = st.text_area(
        "Custom Stopwords (comma or line separated)",
        value="would, could, also, get, use, make, know",
        help="Add domain-specific noise words to filter out.",
    )
    custom_stopwords_list = [
        w.strip().lower() for w in custom_stops_raw.replace("\n", ",").split(",") if w.strip()
    ]

    # 3. MODEL HYPERPARAMETERS
    st.divider()
    st.subheader("3. Model Settings")
    algorithm = st.selectbox(
        "Algorithm",
        ["LDA (Latent Dirichlet Allocation)", "NMF (Non-Negative Matrix Factorization)"],
        index=0,
        help="LDA produces probabilistic generative topics. NMF produces linear parts-based decompositions via TF-IDF.",
    )
    algo_key = "LDA" if "LDA" in algorithm else "NMF"

    n_topics = st.slider("Number of Topics (k)", min_value=2, max_value=15, value=4)
    n_top_words = st.slider("Top Words per Topic", min_value=5, max_value=20, value=10)

    with st.expander("Advanced Vectorizer & Solver Settings", expanded=False):
        max_iter = st.slider("Max Iterations", min_value=10, max_value=100, value=25, step=5)
        max_features = st.slider("Max Vocab Features", min_value=300, max_value=5000, value=1500, step=100)
        max_df = st.slider("Max Document Frequency (max_df)", min_value=0.5, max_value=1.0, value=0.95, step=0.05)
        min_df = st.number_input("Min Document Frequency (min_df)", min_value=1, max_value=10, value=2)
        ngram_choice = st.selectbox("N-Gram Range", ["Unigrams (1, 1)", "Unigrams + Bigrams (1, 2)"], index=0)
        ngram_range = (1, 1) if "1, 1" in ngram_choice else (1, 2)
        random_state = st.number_input("Random Seed (Reproducibility)", value=42)

    # 4. ACTION BUTTONS
    st.divider()
    col_run, col_reset = st.columns([2, 1])

    with col_run:
        run_button = st.button("🚀 Run Modelling", type="primary", use_container_width=True)

    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.clear()
            st.rerun()


# --- EXECUTION LOGIC ---
if run_button:
    if st.session_state.raw_df is None or st.session_state.raw_df.empty:
        st.error("⚠️ No documents available. Please upload files or select sample data from the sidebar.")
    else:
        is_valid, err_msg = validate_documents(st.session_state.raw_df, min_docs=5)
        if not is_valid:
            st.error(f"⚠️ Validation Error: {err_msg}")
        else:
            with st.status("🧠 Processing Documents & Training Topic Model...", expanded=True) as status:
                st.write("1️⃣ Applying text preprocessing and cleaning pipeline...")
                prep_options = PreprocessingOptions(
                    lowercase=opt_lower,
                    remove_urls=opt_urls,
                    remove_emails=opt_emails,
                    remove_html=opt_html,
                    remove_numbers=opt_numbers,
                    remove_punctuation=opt_punct,
                    remove_special=opt_punct,
                    remove_stopwords=opt_stopwords,
                    lemmatize=opt_lemmatize,
                    min_token_len=opt_min_len,
                    custom_stopwords=custom_stopwords_list,
                )
                preprocessor = TextPreprocessor(options=prep_options)
                processed_df, dropped_df = preprocessor.preprocess_dataframe(st.session_state.raw_df)

                if processed_df.empty or len(processed_df) < 2:
                    st.error("All documents were dropped after preprocessing. Try relaxing cleaning filters.")
                    status.update(label="Training failed.", state="error")
                else:
                    st.write(
                        f"✅ Cleaned {len(processed_df)} documents "
                        f"({len(dropped_df)} dropped as empty)."
                    )

                    st.write(f"2️⃣ Training {algo_key} topic model with k={n_topics}...")
                    engine = TopicModelEngine(
                        algorithm=algo_key,
                        n_topics=n_topics,
                        max_iter=max_iter,
                        random_state=random_state,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        ngram_range=ngram_range,
                    )

                    corpus_list = processed_df["cleaned_text"].tolist()
                    doc_topic_mat = engine.fit_transform(corpus_list, n_top_words=n_top_words)

                    st.write("3️⃣ Assigning dominant topics and confidence scores...")
                    assigned_df = engine.assign_documents(processed_df)

                    st.write("4️⃣ Computing model diagnostics and evaluation metrics...")
                    dtm = engine.vectorizer.transform(corpus_list)
                    if algo_key == "LDA":
                        metrics = calculate_lda_metrics(engine.model, dtm)
                    else:
                        metrics = calculate_nmf_metrics(engine.model)

                    # Compute coherence
                    tokenized = processed_df["tokens"].tolist()
                    topic_words = [ts.top_words for ts in engine.topic_summaries]
                    c_score, c_msg = compute_coherence_score(tokenized, topic_words)

                    # Store in session state
                    st.session_state.processed_df = processed_df
                    st.session_state.dropped_df = dropped_df
                    st.session_state.preprocessor = preprocessor
                    st.session_state.model_engine = engine
                    st.session_state.assigned_df = assigned_df
                    st.session_state.lda_metrics = metrics
                    st.session_state.coherence_info = (c_score, c_msg)
                    st.session_state.is_model_trained = True

                    status.update(label="🎉 Topic Modelling Complete!", state="complete", expanded=False)
                    st.toast("Model trained and topics discovered successfully!", icon="✅")


# --- MAIN APPLICATION HEADER ---
st.title("🔬 Text Topic Modelling Studio")
st.caption(
    "Uncover latent topics, analyze keyword importance, group documents by themes, "
    "and evaluate model coherence using state-of-the-art NLP pipelines."
)

if not st.session_state.is_model_trained:
    # Warm onboarding prompt
    st.info(
        "👋 **Welcome!** Select your data source in the sidebar (or use the preloaded Sample Data), "
        "adjust settings if desired, and click **🚀 Run Modelling** to discover topics."
    )

# --- 6 MAIN DASHBOARD TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data Overview",
    "💡 Discovered Topics",
    "📄 Document Assignments",
    "📈 Visualizations",
    "🎯 Model Evaluation",
    "✍️ Try Your Own Text",
])


# ==========================================
# TAB 1: DATA OVERVIEW
# ==========================================
with tab1:
    st.subheader("Data Summary & Inspection")

    if st.session_state.raw_df is not None:
        raw_count = len(st.session_state.raw_df)
        clean_count = len(st.session_state.processed_df) if st.session_state.processed_df is not None else 0
        dropped_count = len(st.session_state.dropped_df) if st.session_state.dropped_df is not None else 0
        vocab_size = len(st.session_state.model_engine.feature_names) if st.session_state.model_engine else "N/A"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw Documents", raw_count)
        c2.metric("Cleaned Documents", clean_count if clean_count > 0 else raw_count)
        c3.metric("Dropped (Empty)", dropped_count)
        c4.metric("Vocabulary Features", vocab_size)

        st.caption(f"**Current Dataset**: {st.session_state.data_source_name}")

        st.divider()
        st.subheader("Document Samples")
        display_df = (
            st.session_state.processed_df if st.session_state.processed_df is not None else st.session_state.raw_df
        )
        st.dataframe(
            display_df[["doc_id", "title", "text" if "text" in display_df.columns else "raw_text"]].head(25),
            use_container_width=True,
            height=280,
        )

        st.divider()
        st.subheader("Before vs. After Cleaning Preview")
        st.caption("Inspect how the text cleaning pipeline strips noise, stopwords, and normalizes lemmas.")

        sample_idx = st.selectbox(
            "Select Document to Compare",
            options=list(range(min(15, len(st.session_state.raw_df)))),
            format_func=lambda i: f"Document {i + 1}: {st.session_state.raw_df.iloc[i].get('title', f'Doc {i+1}')}",
        )

        sample_raw = str(st.session_state.raw_df.iloc[sample_idx].get("text", st.session_state.raw_df.iloc[sample_idx].get("raw_text", "")))
        active_prep = st.session_state.preprocessor or TextPreprocessor()
        comparison = active_prep.get_before_after_sample(sample_raw)

        col_before, col_after = st.columns(2)
        with col_before:
            st.markdown("**Original Raw Document**")
            st.info(comparison["raw_text"])
            st.caption(f"Word count: {comparison['raw_length']} words")

        with col_after:
            st.markdown("**Preprocessed & Lemmatized Tokens**")
            st.success(comparison["cleaned_text"] if comparison["cleaned_text"] else "*(Empty document)*")
            st.caption(f"Cleaned token count: {comparison['cleaned_length']} tokens")

    else:
        st.warning("No dataset loaded yet. Please configure the sidebar and click Run Modelling.")


# ==========================================
# TAB 2: DISCOVERED TOPICS
# ==========================================
with tab2:
    st.subheader("Discovered Latent Topics")

    if st.session_state.is_model_trained and st.session_state.model_engine:
        engine: TopicModelEngine = st.session_state.model_engine
        summaries = engine.topic_summaries

        st.write(
            f"The **{engine.algorithm}** model discovered **{len(summaries)} distinct topics** "
            f"from the corpus."
        )

        # Topic summary pills
        pill_cols = st.columns(len(summaries))
        for idx, summary in enumerate(summaries):
            with pill_cols[idx]:
                st.markdown(
                    f"<div class='metric-card'><b>Topic {idx + 1}</b><br><small>{summary.label.split(': ')[-1]}</small></div>",
                    unsafe_allow_html=True,
                )

        st.write("")
        view_mode = st.radio("View Mode", ["Detailed Topic Inspector", "All Topics Combined Grid"], horizontal=True)

        if view_mode == "Detailed Topic Inspector":
            selected_topic_idx = st.selectbox(
                "Select Topic to Inspect",
                range(len(summaries)),
                format_func=lambda i: summaries[i].label,
            )
            curr_topic = summaries[selected_topic_idx]

            col_chart, col_cloud = st.columns([1.2, 1])

            with col_chart:
                st.markdown(f"#### Top Keywords for {curr_topic.label}")
                fig_bar = plot_top_words_bar(curr_topic)
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_cloud:
                st.markdown("#### Topic Word Cloud")
                word_weights = dict(curr_topic.word_weight_pairs)
                fig_wc = generate_wordcloud_figure(word_weights, title=curr_topic.label)
                st.pyplot(fig_wc)

            st.markdown("##### Word Weights Table")
            words_df = pd.DataFrame(curr_topic.word_weight_pairs, columns=["Keyword", "Normalized Weight"])
            st.dataframe(words_df.T, use_container_width=True)

        else:
            fig_all = plot_all_topics_grid(summaries)
            st.plotly_chart(fig_all, use_container_width=True)

            st.divider()
            st.markdown("#### Word Clouds for All Topics")
            wc_cols = st.columns(min(3, len(summaries)))
            for idx, summary in enumerate(summaries):
                with wc_cols[idx % 3]:
                    word_weights = dict(summary.word_weight_pairs)
                    fig_wc = generate_wordcloud_figure(word_weights, title=summary.label)
                    st.pyplot(fig_wc)

    else:
        st.info("Run topic modelling from the sidebar to inspect discovered topics and word clouds.")


# ==========================================
# TAB 3: DOCUMENT ASSIGNMENTS
# ==========================================
with tab3:
    st.subheader("Document Topic Assignments")

    if st.session_state.is_model_trained and st.session_state.assigned_df is not None:
        assigned_df: pd.DataFrame = st.session_state.assigned_df

        col_filt1, col_filt2, col_down = st.columns([1.5, 1.5, 1])

        with col_filt1:
            topic_filter_options = ["All Topics"] + list(st.session_state.model_engine.topic_labels.values())
            selected_filter = st.selectbox("Filter by Topic", topic_filter_options)

        with col_filt2:
            search_query = st.text_input("🔍 Search Document Text / Snippet", placeholder="Type keyword...")

        filtered_df = assigned_df.copy()
        if selected_filter != "All Topics":
            filtered_df = filtered_df[filtered_df["topic_label"] == selected_filter]

        if search_query.strip():
            filtered_df = filtered_df[
                filtered_df["raw_text"].str.contains(search_query, case=False, na=False)
                | filtered_df["title"].str.contains(search_query, case=False, na=False)
            ]

        with col_down:
            st.write("")
            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="topic_modelling_document_assignments.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.caption(f"Showing **{len(filtered_df)}** of **{len(assigned_df)}** documents")

        # Formatted Display Table
        display_cols = ["doc_id", "title", "snippet", "dominant_topic", "topic_label", "confidence"]
        st.dataframe(
            filtered_df[display_cols].sort_values(by="confidence", ascending=False),
            use_container_width=True,
            height=400,
            column_config={
                "doc_id": "Doc ID",
                "title": "Title / Source",
                "snippet": "Text Snippet",
                "dominant_topic": "Topic #",
                "topic_label": "Assigned Label",
                "confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    help="Model assignment probability score (0 to 1)",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                ),
            },
        )

        with st.expander("Inspect Single Document Full Text", expanded=False):
            inspect_doc_id = st.selectbox(
                "Select Document ID to read in full",
                filtered_df["doc_id"].tolist() if not filtered_df.empty else [0],
            )
            doc_row = assigned_df[assigned_df["doc_id"] == inspect_doc_id]
            if not doc_row.empty:
                st.write(f"**Title**: {doc_row.iloc[0]['title']}")
                st.write(f"**Dominant Topic**: `{doc_row.iloc[0]['topic_label']}` (Confidence: {doc_row.iloc[0]['confidence']:.4f})")
                st.text_area("Full Document Text", value=doc_row.iloc[0]["raw_text"], height=160, disabled=True)

    else:
        st.info("Run topic modelling from the sidebar to view document assignments and download the CSV.")


# ==========================================
# TAB 4: VISUALIZATIONS
# ==========================================
with tab4:
    st.subheader("Corpus & Document Topic Visualizations")

    if st.session_state.is_model_trained and st.session_state.model_engine:
        engine: TopicModelEngine = st.session_state.model_engine
        assigned_df: pd.DataFrame = st.session_state.assigned_df
        doc_topic_mat = engine.doc_topic_matrix

        col_v1, col_v2 = st.columns(2)

        with col_v1:
            st.markdown("#### 1. Corpus Topic Distribution")
            chart_type = st.radio("Chart Type", ["Bar", "Pie"], horizontal=True)
            fig_dist = plot_topic_distribution(assigned_df, plot_type=chart_type)
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_v2:
            st.markdown("#### 2. 2D Document Topic Clustering")
            proj_method = st.radio("Projection Method", ["PCA", "t-SNE"], horizontal=True)
            fig_proj = plot_documents_2d_projection(doc_topic_mat, assigned_df, method=proj_method)
            st.plotly_chart(fig_proj, use_container_width=True)

        st.divider()
        st.markdown("#### 3. Document-Topic Assignment Heatmap")
        st.caption("Inspect topic intensity weights across a sample of documents.")
        sample_heat_count = st.slider("Sample Document Count", min_value=5, max_value=50, value=20)
        topic_labels_list = [summary.label for summary in engine.topic_summaries]
        doc_titles = assigned_df["title"].tolist()

        fig_heat = plot_document_topic_heatmap(
            doc_topic_mat,
            topic_labels=topic_labels_list,
            doc_titles=doc_titles,
            max_docs=sample_heat_count,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    else:
        st.info("Run topic modelling from the sidebar to generate interactive visualizations.")


# ==========================================
# TAB 5: MODEL EVALUATION
# ==========================================
with tab5:
    st.subheader("Model Evaluation & Topic Optimization")

    if st.session_state.is_model_trained and st.session_state.model_engine:
        engine: TopicModelEngine = st.session_state.model_engine
        metrics = st.session_state.lda_metrics or {}
        coherence_score, coherence_msg = st.session_state.coherence_info or (None, "")

        st.markdown("#### Current Model Performance Metrics")
        m_col1, m_col2, m_col3 = st.columns(3)

        if engine.algorithm == "LDA":
            m_col1.metric("Perplexity", metrics.get("perplexity", "N/A"), help="Lower perplexity indicates better generalization.")
            m_col2.metric("Log-Likelihood", metrics.get("log_likelihood", "N/A"), help="Higher log-likelihood indicates better fit.")
        else:
            m_col1.metric("Reconstruction Error", metrics.get("reconstruction_error", "N/A"))
            m_col2.metric("Algorithm", "NMF (TF-IDF)")

        if coherence_score is not None:
            m_col3.metric("Topic Coherence (C_v)", f"{coherence_score:.4f}", help="Higher coherence means words co-occur meaningfully.")
        else:
            m_col3.metric("Topic Coherence (C_v)", "Optional", help=coherence_msg)

        if coherence_msg:
            st.caption(f"ℹ️ {coherence_msg}")

        st.divider()
        st.markdown("#### 🔍 Find Best Number of Topics (k-Search)")
        st.write(
            "Automatically train models across a range of topic counts and evaluate "
            "metrics to identify the optimal semantic balance."
        )

        col_k1, col_k2, col_btn = st.columns([1, 1, 1.2])
        with col_k1:
            k_min = st.number_input("Min Topics (k_min)", min_value=2, max_value=8, value=2)
        with col_k2:
            k_max = st.number_input("Max Topics (k_max)", min_value=3, max_value=15, value=7)

        with col_btn:
            st.write("")
            start_search = st.button("⚡ Run Topic Count Grid Search", use_container_width=True)

        if start_search:
            if k_min >= k_max:
                st.error("k_min must be strictly less than k_max.")
            else:
                progress_bar = st.progress(0.0)
                status_txt = st.empty()

                def update_search_progress(ratio: float, current_k: int):
                    progress_bar.progress(ratio)
                    status_txt.text(f"Evaluating model with k = {current_k} topics...")

                eval_df, rec = search_best_topic_count(
                    corpus_texts=st.session_state.processed_df["cleaned_text"].tolist(),
                    tokenized_docs=st.session_state.processed_df["tokens"].tolist(),
                    algorithm=engine.algorithm,
                    min_topics=int(k_min),
                    max_topics=int(k_max),
                    progress_callback=update_search_progress,
                )

                progress_bar.progress(1.0)
                status_txt.empty()

                st.session_state.eval_search_results = eval_df
                st.session_state.eval_recommendation = rec

        if st.session_state.eval_search_results is not None:
            eval_results_df = st.session_state.eval_search_results
            rec = st.session_state.eval_recommendation

            st.success(
                f"💡 **Recommendation**: Optimal number of topics is **k = {rec['recommended_k']}**! "
                f"*{rec['rationale']}*"
            )

            fig_curves = plot_topic_search_curves(eval_results_df, algorithm=rec["algorithm"])
            st.plotly_chart(fig_curves, use_container_width=True)

            with st.expander("View Grid Search Metrics Table", expanded=False):
                st.dataframe(eval_results_df, use_container_width=True)

    else:
        st.info("Run topic modelling from the sidebar to inspect perplexity, log-likelihood, and coherence.")


# ==========================================
# TAB 6: TRY YOUR OWN TEXT
# ==========================================
with tab6:
    st.subheader("Real-Time Topic Inference Sandbox")
    st.write(
        "Paste any raw article, paragraph, or user review to predict its topic "
        "distribution using the trained model."
    )

    if st.session_state.is_model_trained and st.session_state.model_engine:
        engine: TopicModelEngine = st.session_state.model_engine
        preprocessor: TextPreprocessor = st.session_state.preprocessor or TextPreprocessor()

        default_sample_text = (
            "Autonomous space telescopes and orbital satellite communication constellations "
            "provide deep observations of planetary atmospheres and distant galaxies."
        )

        user_input_text = st.text_area(
            "Input Document Text",
            value=default_sample_text,
            height=120,
            placeholder="Paste your text here...",
        )

        if st.button("🔮 Classify Topic", type="primary"):
            if not user_input_text.strip():
                st.warning("Please enter some text to analyze.")
            else:
                # 1. Clean input
                cleaned_input = preprocessor.clean_text(user_input_text)
                cleaned_tokens = preprocessor.clean_text_to_tokens(user_input_text)

                if not cleaned_input.strip():
                    st.warning("Input text contains only stopwords or filtered characters. No valid tokens found.")
                else:
                    # 2. Predict distribution
                    pred_res = engine.predict_new_text(cleaned_input)

                    st.markdown("---")
                    res_col1, res_col2 = st.columns([1, 1.5])

                    with res_col1:
                        st.markdown("##### Prediction Summary")
                        st.markdown(
                            f"<div class='metric-card'>"
                            f"<h3>Dominant: Topic {pred_res['dominant_topic']}</h3>"
                            f"<p><b>Label</b>: {pred_res['dominant_label']}</p>"
                            f"<p><b>Confidence</b>: {pred_res['confidence'] * 100:.1f}%</p>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        st.write("")
                        st.markdown("**Cleaned Tokens Used for Inference:**")
                        st.caption(f"`{'`, `'.join(cleaned_tokens)}`")

                    with res_col2:
                        st.markdown("##### Topic Probability Distribution")
                        dist_df = pd.DataFrame(pred_res["distribution"])
                        dist_df = dist_df.sort_values(by="probability", ascending=True)

                        fig_pred = px.bar(
                            dist_df,
                            x="probability",
                            y="label",
                            orientation="h",
                            text="probability",
                            labels={"probability": "Probability", "label": "Topic"},
                            template="plotly_white",
                            height=300,
                        )
                        fig_pred.update_layout(
                            margin=dict(l=20, r=20, t=20, b=20),
                            xaxis_range=[0, 1.0],
                        )
                        fig_pred.update_traces(
                            marker_color="#1f77b4",
                            texttemplate="%{text:.2%}",
                            textposition="outside",
                        )
                        st.plotly_chart(fig_pred, use_container_width=True)

    else:
        st.info("Run topic modelling from the sidebar first to enable real-time inference on new text.")
