"""Unit tests for the evaluation module."""

from typing import List

import pytest
from sklearn.feature_extraction.text import CountVectorizer

from src.evaluation import (
    calculate_lda_metrics,
    calculate_nmf_metrics,
    search_best_topic_count,
)
from src.topic_model import TopicModelEngine


@pytest.fixture
def sample_corpus() -> List[str]:
    """Sample corpus for evaluation tests."""
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


def test_lda_metrics_calculation(sample_corpus: List[str]) -> None:
    """Verify perplexity and log-likelihood calculation."""
    engine = TopicModelEngine(algorithm="LDA", n_topics=2, max_iter=15, random_state=42)
    engine.fit_transform(sample_corpus)

    dtm = engine.vectorizer.transform(sample_corpus)
    metrics = calculate_lda_metrics(engine.model, dtm)

    assert "perplexity" in metrics
    assert "log_likelihood" in metrics
    assert metrics["perplexity"] > 0
    assert metrics["log_likelihood"] < 0  # Log-likelihood is negative


def test_search_best_topic_count(sample_corpus: List[str]) -> None:
    """Verify topic-count grid search returns valid evaluations and a recommendation."""
    df_results, rec = search_best_topic_count(
        corpus_texts=sample_corpus,
        algorithm="LDA",
        min_topics=2,
        max_topics=4,
        step=1,
    )

    assert len(df_results) == 3  # k=2, 3, 4
    assert "num_topics" in df_results.columns
    assert "perplexity" in df_results.columns
    assert "recommended_k" in rec
    assert rec["recommended_k"] in [2, 3, 4]
    assert len(rec["rationale"]) > 0
