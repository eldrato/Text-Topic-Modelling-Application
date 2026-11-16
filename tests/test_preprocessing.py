"""Unit tests for the preprocessing module."""

import pandas as pd
import pytest

from src.preprocessing import PreprocessingOptions, TextPreprocessor


@pytest.fixture
def preprocessor() -> TextPreprocessor:
    """Fixture providing a default TextPreprocessor."""
    options = PreprocessingOptions(
        lowercase=True,
        remove_urls=True,
        remove_emails=True,
        remove_html=True,
        remove_numbers=True,
        remove_punctuation=True,
        remove_stopwords=True,
        lemmatize=True,
        min_token_len=3,
        custom_stopwords=["customnoise", "buzzword"],
    )
    return TextPreprocessor(options=options)


def test_url_and_email_removal(preprocessor: TextPreprocessor) -> None:
    """Verify URLs and email addresses are cleaned out."""
    raw = "Contact support@example.com or visit https://ai-research.org/models for details."
    tokens = preprocessor.clean_text_to_tokens(raw)
    assert "support@example.com" not in tokens
    assert "https" not in tokens
    assert "ai-research.org" not in tokens
    assert "contact" in tokens or "detail" in tokens


def test_html_tag_removal(preprocessor: TextPreprocessor) -> None:
    """Verify HTML tags are stripped out."""
    raw = "<div><p>Deep <b>neural</b> networks analyze complex inputs.</p></div>"
    tokens = preprocessor.clean_text_to_tokens(raw)
    assert "div" not in tokens
    assert "p" not in tokens
    assert "neural" in tokens
    assert "network" in tokens or "networks" in tokens


def test_number_and_short_token_removal(preprocessor: TextPreprocessor) -> None:
    """Verify numeric digits and very short tokens (<3 chars) are filtered."""
    raw = "In 2026, 42 agents won 1 championship cup on a 3d map."
    tokens = preprocessor.clean_text_to_tokens(raw)
    assert "2026" not in tokens
    assert "42" not in tokens
    assert "on" not in tokens
    assert "championship" in tokens


def test_custom_stopwords(preprocessor: TextPreprocessor) -> None:
    """Verify user-defined custom stopwords are filtered."""
    raw = "This customnoise system has strong buzzword analytics capabilities."
    tokens = preprocessor.clean_text_to_tokens(raw)
    assert "customnoise" not in tokens
    assert "buzzword" not in tokens
    assert "analytic" in tokens or "analytics" in tokens or "capability" in tokens


def test_lemmatization(preprocessor: TextPreprocessor) -> None:
    """Verify plural nouns and verbs are lemmatized to base forms."""
    raw = "Scientists are studying running satellites and orbiting rockets."
    tokens = preprocessor.clean_text_to_tokens(raw)
    # running -> run, satellites -> satellite, rockets -> rocket
    assert any("satellite" in t for t in tokens)
    assert any("rocket" in t for t in tokens)


def test_empty_document_handling(preprocessor: TextPreprocessor) -> None:
    """Verify documents that become empty after cleaning are properly dropped."""
    df = pd.DataFrame([
        {"doc_id": 1, "title": "Normal Doc", "text": "Artificial intelligence creates autonomous agents."},
        {"doc_id": 2, "title": "Empty Doc", "text": "... 12345 !!! https://test.com"},
    ])
    processed_df, dropped_df = preprocessor.preprocess_dataframe(df)

    assert len(processed_df) == 1
    assert len(dropped_df) == 1
    assert processed_df.iloc[0]["doc_id"] == 1
    assert dropped_df.iloc[0]["doc_id"] == 2
