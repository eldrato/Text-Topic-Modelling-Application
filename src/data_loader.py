"""Data ingestion module for Text Topic Modelling application.

Supports:
- Multiple uploaded .txt files
- CSV/XLSX file uploads with user-selected text column
- Scikit-learn 20 Newsgroups dataset (with subset size and category filters)
- Offline fallback to bundled sample data
- Robust encoding detection and validation
"""

import io
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
from sklearn.datasets import fetch_20newsgroups


AVAILABLE_20NEWS_CATEGORIES = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
]


def decode_bytes(raw_bytes: bytes) -> str:
    """Decode raw bytes trying UTF-8 first, then Latin-1.

    Args:
        raw_bytes: Binary data of a text file.

    Returns:
        Decoded text string.
    """
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return raw_bytes.decode("latin-1")
        except Exception:
            return raw_bytes.decode("utf-8", errors="replace")


def load_uploaded_txts(uploaded_files: List[Union[io.BytesIO, any]]) -> pd.DataFrame:
    """Load multiple uploaded .txt files into a unified DataFrame.

    Args:
        uploaded_files: List of file objects (e.g., Streamlit UploadedFile).

    Returns:
        DataFrame with columns ['doc_id', 'title', 'text'].
    """
    records = []
    for idx, f in enumerate(uploaded_files):
        name = getattr(f, "name", f"doc_{idx + 1}.txt")
        try:
            if hasattr(f, "read"):
                content_bytes = f.read()
                # Reset pointer if possible
                if hasattr(f, "seek"):
                    f.seek(0)
            elif isinstance(f, (str, Path)):
                with open(f, "rb") as fh:
                    content_bytes = fh.read()
            else:
                content_bytes = bytes(f)

            text = decode_bytes(content_bytes).strip()
            if text:
                records.append({
                    "doc_id": idx,
                    "title": name,
                    "text": text,
                })
        except Exception as e:
            # Skip unreadable single file gracefully
            continue

    return pd.DataFrame(records)


def load_csv_or_excel(
    file_obj: Union[io.BytesIO, str, Path],
    filename: str,
    text_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load a CSV or Excel file and return dataframe and available columns.

    Args:
        file_obj: File-like object or path.
        filename: Name of the file to determine extension.
        text_column: Name of the column containing document text (optional).

    Returns:
        Tuple of (DataFrame with ['doc_id', 'title', 'text'], list_of_all_columns).
    """
    lower_name = filename.lower()
    if lower_name.endswith(".csv"):
        try:
            df = pd.read_csv(file_obj, encoding="utf-8")
        except UnicodeDecodeError:
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            df = pd.read_csv(file_obj, encoding="latin-1")
    elif lower_name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_obj)
    else:
        raise ValueError(f"Unsupported file format for {filename}. Please upload a CSV or XLSX file.")

    columns = list(df.columns)
    if not text_column:
        # Default to first string column or first column
        text_cols = [c for c in columns if df[c].dtype == object]
        text_column = text_cols[0] if text_cols else columns[0]

    if text_column not in df.columns:
        raise KeyError(f"Selected text column '{text_column}' not found in file columns: {columns}")

    # Clean missing values
    valid_df = df[df[text_column].notna()].copy()
    valid_df[text_column] = valid_df[text_column].astype(str).str.strip()
    valid_df = valid_df[valid_df[text_column].str.len() > 0]

    records = []
    for idx, row in valid_df.reset_index(drop=True).iterrows():
        title_val = f"Doc {idx + 1}"
        if "title" in row and pd.notna(row["title"]):
            title_val = str(row["title"])
        elif "id" in row and pd.notna(row["id"]):
            title_val = f"ID: {row['id']}"

        records.append({
            "doc_id": idx,
            "title": title_val,
            "text": str(row[text_column]),
        })

    return pd.DataFrame(records), columns


def load_20newsgroups(
    subset_size: int = 1000,
    categories: Optional[List[str]] = None,
    random_state: int = 42,
    base_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, bool]:
    """Fetch 20 Newsgroups documents with offline fallback.

    Args:
        subset_size: Maximum number of documents to retrieve.
        categories: Optional list of category names to filter.
        random_state: Seed for reproducibility.
        base_dir: Optional base path to locate sample_data directory.

    Returns:
        Tuple of (DataFrame with ['doc_id', 'title', 'text'], is_offline_fallback).
    """
    try:
        data = fetch_20newsgroups(
            subset="all",
            categories=categories if categories else None,
            shuffle=True,
            random_state=random_state,
            remove=("headers", "footers", "quotes"),
        )
        records = []
        for i, text in enumerate(data.data):
            cleaned = text.strip()
            if len(cleaned) >= 20:  # Skip trivial snippets
                records.append({
                    "doc_id": len(records),
                    "title": f"20news_{data.target_names[data.target[i]]}_{i}",
                    "text": cleaned,
                })
            if len(records) >= subset_size:
                break

        if len(records) >= 10:
            return pd.DataFrame(records), False

    except Exception:
        # Network unavailable or download failure -> fallback
        pass

    # Offline fallback
    fallback_df = load_sample_fallback(base_dir=base_dir)
    return fallback_df, True


def load_sample_fallback(base_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load bundled sample data from sample_data/ directory.

    Args:
        base_dir: Root directory of the application.

    Returns:
        DataFrame with sample documents.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent

    sample_dir = base_dir / "sample_data"
    txt_dir = sample_dir / "txt_docs"

    records = []
    # 1. Try reading individual txt files
    if txt_dir.exists():
        txt_files = sorted(txt_dir.glob("*.txt"))
        for idx, file_path in enumerate(txt_files):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    records.append({
                        "doc_id": len(records),
                        "title": file_path.name,
                        "text": content,
                    })
            except Exception:
                continue

    # 2. Try reading sample feedback.csv if txt files are insufficient
    feedback_csv = sample_dir / "feedback.csv"
    if feedback_csv.exists():
        try:
            csv_df = pd.read_csv(feedback_csv)
            for idx, row in csv_df.iterrows():
                text_val = str(row.get("text", row.get("review", ""))).strip()
                if text_val:
                    records.append({
                        "doc_id": len(records),
                        "title": f"Feedback_{row.get('id', idx + 1)}",
                        "text": text_val,
                    })
        except Exception:
            pass

    if not records:
        # Hardcoded emergency fallback in case files are missing
        hardcoded_docs = [
            "Artificial intelligence and machine learning models are transforming automated software engineering.",
            "Deep neural networks require large datasets and GPU compute clusters to optimize loss functions.",
            "NASA launched a new rover to explore Martian geological formations and search for past microbial life.",
            "Astronomers detected distant exoplanets orbiting habitable zones using space telescope photometry.",
            "Solar energy installations and wind turbine farms accelerate the transition toward renewable electricity.",
            "Lithium-ion battery storage facilities stabilize national power grids during peak electrical demand.",
            "Cardiovascular exercise and balanced Mediterranean diets significantly decrease arterial disease risks.",
            "Clinical pharmacology studies investigate antibody therapies for targeted immunological treatments.",
            "Central banks regulate interest rates to mitigate runaway inflation while supporting labor employment.",
            "Sovereign bond yields fluctuated following quarterly macroeconomic reports and GDP estimates.",
            "Premier league football clubs compete intensely for the international championship tournament trophy.",
            "Olympic sprint athletes train rigorous interval workouts to maximize peak anaerobic muscular velocity.",
        ]
        records = [
            {"doc_id": i, "title": f"sample_doc_{i + 1}.txt", "text": t}
            for i, t in enumerate(hardcoded_docs)
        ]

    return pd.DataFrame(records)


def validate_documents(
    df: pd.DataFrame,
    text_column: str = "text",
    min_docs: int = 10,
) -> Tuple[bool, str]:
    """Validate that the documents meet minimum requirements.

    Args:
        df: Input DataFrame.
        text_column: Column name to evaluate.
        min_docs: Minimum required document count (default 10).

    Returns:
        Tuple of (is_valid: bool, error_message: str).
    """
    if df is None or df.empty:
        return False, "No documents loaded. Please provide valid text input."

    if text_column not in df.columns:
        return False, f"Missing required text column '{text_column}' in the dataset."

    # Filter out empty or whitespace-only documents
    non_empty = df[df[text_column].astype(str).str.strip().str.len() > 0]
    count = len(non_empty)

    if count < min_docs:
        return (
            False,
            f"Dataset contains only {count} valid document(s). Topic modelling requires at least "
            f"{min_docs} documents to discover meaningful latent patterns.",
        )

    return True, ""
