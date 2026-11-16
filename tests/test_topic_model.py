"""Unit tests for the topic modelling engine."""

from typing import List

import numpy as np
import pandas as pd
import pytest

from src.topic_model import TopicModelEngine


@pytest.fixture
def sample_corpus() -> List[str]:
    """Sample preprocessed documents covering two clear topics (space vs finance)."""
    return [
        "space telescope orbit satellite planet atmosphere galaxy",
        "nasa rover mars mission astronaut lunar rocket landing",
        "astronomy observatory planetary interstellar solar telescope",
        "satellite trajectory orbit lunar rocket booster deep space",
        "central bank interest rate inflation economic currency monetary",
        "stock equity market treasury bond hedge fund portfolio",
        "macroeconomic monetary inflation policy gdp growth banking",
        "financial equity exchange dividend asset yield capital",
        "space mission telescope orbit planetary exploration astronomy",
        "interest rate monetary inflation central bank treasury bond",
    ]


def test_lda_fit_transform_and_normalization(sample_corpus: List[str]) -> None:
    """Verify LDA trains and outputs a normalized probability matrix."""
    engine = TopicModelEngine(algorithm="LDA", n_topics=2, max_iter=20, random_state=42)
    doc_topic_matrix = engine.fit_transform(sample_corpus, n_top_words=5)

    assert doc_topic_matrix.shape == (len(sample_corpus), 2)
    # Each document's topic distribution should sum to ~1.0
    row_sums = doc_topic_matrix.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(len(sample_corpus)), atol=1e-4)


def test_nmf_fit_transform_and_normalization(sample_corpus: List[str]) -> None:
    """Verify NMF trains and outputs normalized row vectors."""
    engine = TopicModelEngine(algorithm="NMF", n_topics=2, max_iter=100, random_state=42)
    doc_topic_matrix = engine.fit_transform(sample_corpus, n_top_words=5)

    assert doc_topic_matrix.shape == (len(sample_corpus), 2)
    row_sums = doc_topic_matrix.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(len(sample_corpus)), atol=1e-4)


def test_top_words_and_labels(sample_corpus: List[str]) -> None:
    """Verify top words and auto-generated labels are produced correctly."""
    engine = TopicModelEngine(algorithm="LDA", n_topics=2, max_iter=20, random_state=42)
    engine.fit_transform(sample_corpus, n_top_words=6)

    summaries = engine.topic_summaries
    assert len(summaries) == 2

    for summary in summaries:
        assert len(summary.top_words) == 6
        assert len(summary.top_weights) == 6
        assert ":" in summary.label
        assert len(summary.word_weight_pairs) == 6


def test_document_assignment(sample_corpus: List[str]) -> None:
    """Verify document assignment assigns dominant topics and valid confidence scores."""
    engine = TopicModelEngine(algorithm="LDA", n_topics=2, max_iter=20, random_state=42)
    engine.fit_transform(sample_corpus, n_top_words=5)

    df = pd.DataFrame({
        "doc_id": list(range(len(sample_corpus))),
        "title": [f"Doc_{i}" for i in range(len(sample_corpus))],
        "raw_text": sample_corpus,
        "cleaned_text": sample_corpus,
    })

    assigned_df = engine.assign_documents(df)

    assert "dominant_topic" in assigned_df.columns
    assert "confidence" in assigned_df.columns
    assert "topic_label" in assigned_df.columns
    assert (assigned_df["confidence"] >= 0.0).all()
    assert (assigned_df["confidence"] <= 1.0).all()
    assert set(assigned_df["dominant_topic"].unique()).issubset({1, 2})


def test_inference_on_new_text(sample_corpus: List[str]) -> None:
    """Verify inference produces valid predictions for unseen text."""
    engine = TopicModelEngine(algorithm="LDA", n_topics=2, max_iter=20, random_state=42)
    engine.fit_transform(sample_corpus, n_top_words=5)

    new_text = "space rocket orbit mission"
    pred = engine.predict_new_text(new_text)

    assert "dominant_topic" in pred
    assert "dominant_label" in pred
    assert "confidence" in pred
    assert "distribution" in pred
    assert len(pred["distribution"]) == 2
    assert pred["dominant_topic"] in [1, 2]
    assert 0.0 <= pred["confidence"] <= 1.0
