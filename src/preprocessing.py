"""Text preprocessing pipeline for Text Topic Modelling.

Includes:
- Auto-downloading required NLTK resources on first run (with offline support)
- Lowercasing, URL, email, HTML, number, and punctuation removal
- Word tokenization and stopword removal (with user-defined custom stopwords)
- Lemmatization using WordNetLemmatizer
- Minimum token length filtering
- Empty document pruning while tracking original document IDs
- Before vs. after comparison utility
"""

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Set, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd


# Default fallback English stopwords in case NLTK resources are unavailable offline
BUILTIN_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "also", "would", "could", "should", "like", "one",
    "get", "use", "make", "said", "just", "know", "take", "time", "year", "people",
}


_NLTK_RESOURCES_INITIALIZED = False


def ensure_nltk_resources() -> None:
    """Download required NLTK resources quietly; fall back gracefully if offline."""
    global _NLTK_RESOURCES_INITIALIZED
    if _NLTK_RESOURCES_INITIALIZED:
        return

    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                # If network fails, ignore and use fallback tokenizers and stopwords
                pass

    _NLTK_RESOURCES_INITIALIZED = True


@dataclass
class PreprocessingOptions:
    """Configurable options for the text preprocessing pipeline."""

    lowercase: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_html: bool = True
    remove_numbers: bool = True
    remove_punctuation: bool = True
    remove_special: bool = True
    remove_stopwords: bool = True
    lemmatize: bool = True
    min_token_len: int = 3
    custom_stopwords: List[str] = field(default_factory=list)


class TextPreprocessor:
    """Pipeline for cleaning, tokenizing, filtering, and lemmatizing text."""

    def __init__(self, options: Optional[PreprocessingOptions] = None):
        """Initialize preprocessor with options and NLTK resources."""
        ensure_nltk_resources()
        self.options = options or PreprocessingOptions()

        # Initialize lemmatizer
        try:
            self.lemmatizer = WordNetLemmatizer()
            # Test lemmatizer
            _ = self.lemmatizer.lemmatize("running", pos="v")
        except Exception:
            self.lemmatizer = None

        # Build stopword set
        self.stop_words: Set[str] = set()
        if self.options.remove_stopwords:
            try:
                self.stop_words.update(stopwords.words("english"))
            except Exception:
                self.stop_words.update(BUILTIN_STOPWORDS)

        if self.options.custom_stopwords:
            cleaned_custom = [
                w.strip().lower() for w in self.options.custom_stopwords if w.strip()
            ]
            self.stop_words.update(cleaned_custom)

        # Precompiled regex patterns for speed
        self._url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self._email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
        self._html_pattern = re.compile(r"<.*?>")
        self._number_pattern = re.compile(r"\b\d+\b")
        self._punct_special_pattern = re.compile(r"[^\w\s]")
        self._whitespace_pattern = re.compile(r"\s+")

    def clean_text_to_tokens(self, text: str) -> List[str]:
        """Transform a raw text document into a list of cleaned tokens.

        Args:
            text: Raw document text string.

        Returns:
            List of filtered, lemmatized tokens.
        """
        if not text or not isinstance(text, str):
            return []

        # 1. Lowercase
        if self.options.lowercase:
            text = text.lower()

        # 2. Remove URLs
        if self.options.remove_urls:
            text = self._url_pattern.sub(" ", text)

        # 3. Remove Emails
        if self.options.remove_emails:
            text = self._email_pattern.sub(" ", text)

        # 4. Remove HTML tags
        if self.options.remove_html:
            text = self._html_pattern.sub(" ", text)

        # 5. Remove standalone numbers
        if self.options.remove_numbers:
            text = self._number_pattern.sub(" ", text)

        # 6. Remove punctuation and special characters
        if self.options.remove_punctuation or self.options.remove_special:
            text = self._punct_special_pattern.sub(" ", text)

        # 7. Normalize whitespaces
        text = self._whitespace_pattern.sub(" ", text).strip()

        # 8. Tokenization
        tokens: List[str]
        try:
            tokens = word_tokenize(text)
        except Exception:
            # Fallback simple split if punkt tokenizer has issue
            tokens = text.split()

        cleaned_tokens: List[str] = []
        for token in tokens:
            token = token.strip()
            # 9. Minimum length check
            if len(token) < self.options.min_token_len:
                continue

            # Check if token contains non-alphabetic chars if numbers removed
            if self.options.remove_numbers and any(char.isdigit() for char in token):
                continue

            # 10. Stopwords removal
            if self.options.remove_stopwords and token in self.stop_words:
                continue

            # 11. Lemmatization
            if self.options.lemmatize and self.lemmatizer:
                try:
                    # Lemmatize as noun, then verb for better base form
                    noun_lem = self.lemmatizer.lemmatize(token, pos="n")
                    token = self.lemmatizer.lemmatize(noun_lem, pos="v")
                except Exception:
                    pass

            cleaned_tokens.append(token)

        return cleaned_tokens

    def clean_text(self, text: str) -> str:
        """Clean raw text and return space-joined cleaned string."""
        tokens = self.clean_text_to_tokens(text)
        return " ".join(tokens)

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess an entire DataFrame of documents.

        Args:
            df: Input DataFrame containing documents.
            text_column: Name of the text column.

        Returns:
            Tuple of:
            - processed_df: Cleaned dataframe with non-empty documents and columns
              ['doc_id', 'title', 'raw_text', 'cleaned_text', 'tokens', 'token_count']
            - dropped_df: Records dropped because they were empty after cleaning.
        """
        processed_records = []
        dropped_records = []

        for _, row in df.iterrows():
            doc_id = row.get("doc_id", len(processed_records))
            title = row.get("title", f"Doc {doc_id}")
            raw = str(row.get(text_column, ""))

            tokens = self.clean_text_to_tokens(raw)
            cleaned_str = " ".join(tokens)

            record = {
                "doc_id": doc_id,
                "title": title,
                "raw_text": raw,
                "cleaned_text": cleaned_str,
                "tokens": tokens,
                "token_count": len(tokens),
            }

            if len(tokens) > 0:
                processed_records.append(record)
            else:
                dropped_records.append(record)

        processed_df = pd.DataFrame(processed_records)
        dropped_df = pd.DataFrame(dropped_records)

        return processed_df, dropped_df

    def get_before_after_sample(
        self,
        raw_text: str,
    ) -> Dict[str, any]:
        """Generate a comparison structure between raw text and cleaned tokens."""
        tokens = self.clean_text_to_tokens(raw_text)
        cleaned_text = " ".join(tokens)
        return {
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "raw_length": len(raw_text.split()),
            "cleaned_length": len(tokens),
            "tokens": tokens[:30],  # preview first 30 tokens
        }
