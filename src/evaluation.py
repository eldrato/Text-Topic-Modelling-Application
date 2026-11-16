"""Model evaluation module for topic modelling.

Provides:
- Perplexity and log-likelihood calculation for LDA
- Reconstruction error calculation for NMF
- Topic Coherence score (c_v) using gensim (with safe fallback if gensim is absent)
- Topic-count parameter search over a range of k values with recommendation logic
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def calculate_lda_metrics(
    lda_model: LatentDirichletAllocation,
    dtm: Any,
) -> Dict[str, float]:
    """Calculate perplexity and log-likelihood for an LDA model.

    Args:
        lda_model: Fitted LatentDirichletAllocation instance.
        dtm: Document-term matrix used for evaluation.

    Returns:
        Dict containing 'perplexity' and 'log_likelihood'.
    """
    try:
        perplexity = float(np.round(lda_model.perplexity(dtm), 2))
        log_likelihood = float(np.round(lda_model.score(dtm), 2))
    except Exception as e:
        perplexity = 0.0
        log_likelihood = 0.0

    return {
        "perplexity": perplexity,
        "log_likelihood": log_likelihood,
    }


def calculate_nmf_metrics(
    nmf_model: NMF,
) -> Dict[str, float]:
    """Calculate reconstruction error for an NMF model.

    Args:
        nmf_model: Fitted NMF instance.

    Returns:
        Dict containing 'reconstruction_error'.
    """
    error = float(getattr(nmf_model, "reconstruction_err_", 0.0))
    return {
        "reconstruction_error": float(np.round(error, 4)),
    }


def compute_coherence_score(
    tokenized_docs: List[List[str]],
    top_words_per_topic: List[List[str]],
    coherence: str = "c_v",
) -> Tuple[Optional[float], str]:
    """Calculate topic coherence score using gensim if installed.

    Args:
        tokenized_docs: List of tokenized documents (list of word lists).
        top_words_per_topic: List of top word lists for each topic.
        coherence: Coherence metric name (default 'c_v').

    Returns:
        Tuple of (score_float_or_None, status_message).
    """
    try:
        from gensim.corpora import Dictionary
        from gensim.models.coherencemodel import CoherenceModel

        # Filter out empty docs
        valid_docs = [doc for doc in tokenized_docs if len(doc) > 0]
        if not valid_docs:
            return None, "All documents were empty; cannot compute coherence."

        dictionary = Dictionary(valid_docs)
        # Filter extremes for stability
        dictionary.filter_extremes(no_below=1, no_above=0.95)

        # Ensure topics are within dictionary
        coherence_model = CoherenceModel(
            topics=top_words_per_topic,
            texts=valid_docs,
            dictionary=dictionary,
            coherence=coherence,
        )
        score = float(np.round(coherence_model.get_coherence(), 4))
        return score, f"Calculated {coherence} coherence score: {score:.4f}"
    except ImportError:
        return (
            None,
            "Gensim library is not installed in the environment. "
            "Install `gensim` to enable c_v topic coherence calculation.",
        )
    except Exception as e:
        return None, f"Coherence score skipped: {str(e)}"


def search_best_topic_count(
    corpus_texts: List[str],
    tokenized_docs: Optional[List[List[str]]] = None,
    algorithm: str = "LDA",
    min_topics: int = 2,
    max_topics: int = 10,
    step: int = 1,
    random_state: int = 42,
    progress_callback: Optional[Callable[[float, int], None]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Grid search over a range of topic counts to evaluate metrics and recommend the optimal k.

    Args:
        corpus_texts: List of preprocessed document strings.
        tokenized_docs: Optional list of token lists for coherence calculation.
        algorithm: 'LDA' or 'NMF'.
        min_topics: Minimum k to evaluate.
        max_topics: Maximum k to evaluate.
        step: Step size between k values.
        random_state: Random state for model initialization.
        progress_callback: Optional callback func(progress_ratio, current_k).

    Returns:
        Tuple of (evaluation_results_df, recommendation_dict).
    """
    algorithm = algorithm.upper()
    topic_range = list(range(min_topics, max_topics + 1, step))
    total_steps = len(topic_range)

    # Vectorization
    if algorithm == "NMF":
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=1, max_features=1500)
    else:
        vectorizer = CountVectorizer(max_df=0.95, min_df=1, max_features=1500)

    dtm = vectorizer.fit_transform(corpus_texts)
    feature_names = vectorizer.get_feature_names_out().tolist()

    results = []

    for idx, k in enumerate(topic_range):
        if progress_callback:
            progress_callback((idx + 1) / total_steps, k)

        row: Dict[str, Any] = {"num_topics": k}

        if algorithm == "NMF":
            model = NMF(
                n_components=k,
                max_iter=150,
                random_state=random_state,
                init="nndsvda",
            )
            model.fit(dtm)
            row["reconstruction_error"] = float(np.round(model.reconstruction_err_, 4))
        else:
            model = LatentDirichletAllocation(
                n_components=k,
                max_iter=15,
                learning_method="batch",
                random_state=random_state,
                evaluate_every=0,
            )
            model.fit(dtm)
            row["perplexity"] = float(np.round(model.perplexity(dtm), 2))
            row["log_likelihood"] = float(np.round(model.score(dtm), 2))

        # Check coherence if tokens provided
        if tokenized_docs:
            top_words_per_topic = []
            for comp in model.components_:
                top_indices = comp.argsort()[::-1][:10]
                top_words = [feature_names[i] for i in top_indices]
                top_words_per_topic.append(top_words)

            coherence_val, _ = compute_coherence_score(tokenized_docs, top_words_per_topic)
            if coherence_val is not None:
                row["coherence_c_v"] = coherence_val

        results.append(row)

    df_results = pd.DataFrame(results)

    # Determine recommendation
    rec_k = min_topics
    rationale = ""

    if "coherence_c_v" in df_results.columns and df_results["coherence_c_v"].notna().any():
        # Pick topic count with peak coherence
        best_row = df_results.loc[df_results["coherence_c_v"].idxmax()]
        rec_k = int(best_row["num_topics"])
        rationale = (
            f"Topic count k = {rec_k} achieved the highest semantic coherence "
            f"(C_v = {best_row['coherence_c_v']:.4f})."
        )
    elif algorithm == "LDA" and "perplexity" in df_results.columns:
        # Perplexity elbow / minimum
        min_perp_row = df_results.loc[df_results["perplexity"].idxmin()]
        rec_k = int(min_perp_row["num_topics"])
        rationale = (
            f"Topic count k = {rec_k} yielded the lowest model perplexity "
            f"({min_perp_row['perplexity']:.2f}) and best generalizability on this corpus."
        )
    elif algorithm == "NMF" and "reconstruction_error" in df_results.columns:
        # NMF reconstruction error decreases with k; pick a balanced mid-point or elbow
        rec_k = int(df_results.iloc[len(df_results) // 2]["num_topics"])
        rationale = (
            f"Topic count k = {rec_k} provides a balanced low reconstruction error "
            f"without over-fragmenting topics."
        )

    recommendation = {
        "recommended_k": rec_k,
        "rationale": rationale,
        "algorithm": algorithm,
    }

    return df_results, recommendation
