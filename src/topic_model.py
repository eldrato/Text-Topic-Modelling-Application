"""Topic modelling engine implementing Latent Dirichlet Allocation (LDA) and Non-Negative Matrix Factorization (NMF).

Provides:
- CountVectorizer and TfidfVectorizer vectorization
- LDA and NMF model training
- Extraction of top words with normalized weights
- Auto-generated topic labels
- Document-topic distribution calculation
- Dominant topic and confidence assignment
- Inference for new, unseen text documents
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


@dataclass
class TopicSummary:
    """Structure holding the summary of an individual discovered topic."""

    topic_id: int
    label: str
    top_words: List[str]
    top_weights: List[float]
    word_weight_pairs: List[Tuple[str, float]]


class TopicModelEngine:
    """Topic modelling orchestrator for LDA and NMF models."""

    def __init__(
        self,
        algorithm: str = "LDA",
        n_topics: int = 5,
        max_iter: int = 25,
        learning_method: str = "batch",
        random_state: int = 42,
        max_df: float = 0.95,
        min_df: Union[int, float] = 2,
        max_features: int = 1500,
        ngram_range: Tuple[int, int] = (1, 1),
    ):
        """Initialize the topic modelling engine with algorithm and hyperparameter settings."""
        self.algorithm = algorithm.upper()
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.learning_method = learning_method
        self.random_state = random_state
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.ngram_range = ngram_range

        self.vectorizer: Optional[Union[CountVectorizer, TfidfVectorizer]] = None
        self.model: Optional[Union[LatentDirichletAllocation, NMF]] = None
        self.feature_names: List[str] = []
        self.doc_topic_matrix: Optional[np.ndarray] = None
        self.topic_summaries: List[TopicSummary] = []
        self.topic_labels: Dict[int, str] = {}

    def fit_transform(
        self,
        corpus: List[str],
        n_top_words: int = 10,
    ) -> np.ndarray:
        """Fit vectorizer and topic model, returning document-topic matrix.

        Args:
            corpus: List of preprocessed text documents.
            n_top_words: Number of top descriptive words to extract per topic.

        Returns:
            Normalized document-topic probability/weight matrix (n_docs, n_topics).
        """
        # 1. Vectorization
        # Adapt min_df if corpus is smaller than min_df threshold
        actual_min_df = self.min_df
        if isinstance(actual_min_df, int) and len(corpus) < actual_min_df:
            actual_min_df = 1

        if self.algorithm == "NMF":
            self.vectorizer = TfidfVectorizer(
                max_df=self.max_df,
                min_df=actual_min_df,
                max_features=self.max_features,
                ngram_range=self.ngram_range,
            )
        else:
            self.vectorizer = CountVectorizer(
                max_df=self.max_df,
                min_df=actual_min_df,
                max_features=self.max_features,
                ngram_range=self.ngram_range,
            )

        dtm = self.vectorizer.fit_transform(corpus)
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()

        # Handle edge case where vocab is smaller than requested topics
        actual_topics = min(self.n_topics, max(2, dtm.shape[1]))
        if actual_topics != self.n_topics:
            self.n_topics = actual_topics

        # 2. Model Fitting
        if self.algorithm == "NMF":
            self.model = NMF(
                n_components=self.n_topics,
                max_iter=self.max_iter,
                random_state=self.random_state,
                init="nndsvda",
            )
            raw_doc_topic = self.model.fit_transform(dtm)
            # Normalize rows to sum to 1.0 (probabilistic interpretation)
            row_sums = raw_doc_topic.sum(axis=1, keepdims=True)
            # Avoid division by zero
            row_sums[row_sums == 0] = 1.0
            self.doc_topic_matrix = raw_doc_topic / row_sums
        else:
            self.model = LatentDirichletAllocation(
                n_components=self.n_topics,
                max_iter=self.max_iter,
                learning_method=self.learning_method,
                random_state=self.random_state,
                evaluate_every=0,
            )
            self.doc_topic_matrix = self.model.fit_transform(dtm)

        # 3. Extract top words & labels
        self.extract_topic_summaries(n_top_words=n_top_words)

        return self.doc_topic_matrix

    def extract_topic_summaries(self, n_top_words: int = 10) -> List[TopicSummary]:
        """Extract top words, normalized weights, and generate short topic labels."""
        if self.model is None or not self.feature_names:
            return []

        summaries = []
        labels = {}

        components = self.model.components_
        for topic_idx, topic_weights in enumerate(components):
            # Sort word indices by weight descending
            top_word_indices = topic_weights.argsort()[::-1][:n_top_words]
            top_words = [self.feature_names[i] for i in top_word_indices]
            raw_weights = topic_weights[top_word_indices]

            # Normalize weights for visualization
            sum_weights = raw_weights.sum()
            norm_weights = (raw_weights / sum_weights).tolist() if sum_weights > 0 else [0.0] * len(raw_weights)

            # Auto-generated label: join top 3 words
            label_prefix = f"Topic {topic_idx + 1}"
            top_3_words = top_words[:3]
            short_descriptor = "_".join(top_3_words) if top_3_words else "unlabeled"
            full_label = f"{label_prefix}: {short_descriptor}"

            labels[topic_idx] = full_label

            pairs = list(zip(top_words, norm_weights))
            summary = TopicSummary(
                topic_id=topic_idx,
                label=full_label,
                top_words=top_words,
                top_weights=norm_weights,
                word_weight_pairs=pairs,
            )
            summaries.append(summary)

        self.topic_summaries = summaries
        self.topic_labels = labels
        return summaries

    def assign_documents(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Assign dominant topic and confidence score to each document.

        Args:
            df: DataFrame containing at least ['doc_id', 'title', 'raw_text'].

        Returns:
            Enriched DataFrame with dominant topic assignments.
        """
        if self.doc_topic_matrix is None:
            raise ValueError("Model has not been trained yet. Call fit_transform first.")

        assigned_df = df.copy()

        # Dominant topic index is argmax across columns
        dominant_topics = np.argmax(self.doc_topic_matrix, axis=1)
        # Confidence score is maximum probability
        confidences = np.max(self.doc_topic_matrix, axis=1)

        assigned_df["dominant_topic"] = dominant_topics + 1  # 1-indexed for user readability
        assigned_df["dominant_topic_id"] = dominant_topics   # 0-indexed internal ID
        assigned_df["confidence"] = np.round(confidences, 4)
        assigned_df["topic_label"] = [self.topic_labels.get(t, f"Topic {t + 1}") for t in dominant_topics]

        # Text snippet (first 140 chars)
        assigned_df["snippet"] = assigned_df["raw_text"].apply(
            lambda s: (s[:137] + "...") if len(s) > 140 else s
        )

        return assigned_df

    def predict_new_text(
        self,
        cleaned_text: str,
    ) -> Dict[str, Any]:
        """Infer the topic distribution for a single new document.

        Args:
            cleaned_text: Cleaned text string of the unseen document.

        Returns:
            Dictionary with topic probabilities, dominant topic, and label.
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model has not been trained yet.")

        dtm_new = self.vectorizer.transform([cleaned_text])

        if self.algorithm == "NMF":
            raw_dist = self.model.transform(dtm_new)
            row_sum = raw_dist.sum()
            topic_probs = (raw_dist / row_sum).flatten() if row_sum > 0 else np.zeros(self.n_topics)
        else:
            topic_probs = self.model.transform(dtm_new).flatten()

        dominant_id = int(np.argmax(topic_probs))
        confidence = float(topic_probs[dominant_id])

        distribution = [
            {
                "topic_id": i + 1,
                "label": self.topic_labels.get(i, f"Topic {i + 1}"),
                "probability": float(np.round(p, 4)),
            }
            for i, p in enumerate(topic_probs)
        ]

        return {
            "dominant_topic": dominant_id + 1,
            "dominant_label": self.topic_labels.get(dominant_id, f"Topic {dominant_id + 1}"),
            "confidence": np.round(confidence, 4),
            "distribution": distribution,
        }
