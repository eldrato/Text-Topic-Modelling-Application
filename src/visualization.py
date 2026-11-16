"""Visualization suite for Text Topic Modelling.

Includes:
- Horizontal bar charts of top words and weights per topic (Plotly)
- High-resolution WordClouds per topic (WordCloud + Matplotlib)
- Document topic distribution (Plotly Bar & Pie)
- Document-topic distribution heatmap (Plotly Heatmap)
- 2D PCA & t-SNE scatter projection of documents colored by topic (Plotly Scatter)
- Topic count tuning metric curves (Plotly Line chart)
- Optional pyLDAvis HTML preparation
"""

import io
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from wordcloud import WordCloud


def plot_top_words_bar(
    topic_summary: Any,
    color_scale: str = "Blues",
) -> go.Figure:
    """Generate a horizontal bar chart of the top words for an individual topic.

    Args:
        topic_summary: TopicSummary object with top_words and top_weights.
        color_scale: Color palette name.

    Returns:
        Plotly Figure.
    """
    words = list(reversed(topic_summary.top_words))
    weights = list(reversed(topic_summary.top_weights))

    fig = go.Figure(
        go.Bar(
            x=weights,
            y=words,
            orientation="h",
            marker=dict(
                color=weights,
                colorscale=color_scale,
                showscale=False,
            ),
            text=[f"{w:.3f}" for w in weights],
            textposition="auto",
        )
    )

    fig.update_layout(
        title=f"<b>{topic_summary.label}</b> - Top Keywords & Relative Weights",
        xaxis_title="Normalized Weight",
        yaxis_title="Keywords",
        margin=dict(l=20, r=20, t=40, b=20),
        height=380,
        template="plotly_white",
    )
    return fig


def plot_all_topics_grid(
    topic_summaries: List[Any],
    selected_topic_id: Optional[int] = None,
) -> go.Figure:
    """Plot horizontal bar chart for a selected topic or combined overview."""
    if selected_topic_id is not None and 0 <= selected_topic_id < len(topic_summaries):
        target = topic_summaries[selected_topic_id]
        return plot_top_words_bar(target)

    # Combined bar chart comparing top words across topics
    records = []
    for summary in topic_summaries:
        for w, wt in zip(summary.top_words[:8], summary.top_weights[:8]):
            records.append({
                "Topic": summary.label,
                "Word": w,
                "Weight": wt,
            })
    df = pd.DataFrame(records)

    fig = px.bar(
        df,
        x="Weight",
        y="Word",
        color="Topic",
        facet_col="Topic",
        facet_col_wrap=2,
        orientation="h",
        title="Top Words Across Topics",
        template="plotly_white",
        height=300 * ((len(topic_summaries) + 1) // 2),
    )
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig


def generate_wordcloud_figure(
    word_weight_dict: Dict[str, float],
    title: str = "",
    colormap: str = "viridis",
    width: int = 600,
    height: int = 350,
) -> plt.Figure:
    """Generate a clean Matplotlib figure containing a WordCloud from term weights.

    Args:
        word_weight_dict: Mapping of term to relative weight.
        title: Title for the cloud.
        colormap: Matplotlib colormap name.
        width: Image width.
        height: Image height.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=120)
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    if not word_weight_dict:
        ax.text(0.5, 0.5, "No words available", ha="center", va="center")
        ax.axis("off")
        return fig

    wc = WordCloud(
        width=width,
        height=height,
        background_color="#ffffff",
        colormap=colormap,
        prefer_horizontal=0.85,
        min_font_size=10,
        max_words=60,
    ).generate_from_frequencies(word_weight_dict)

    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    plt.tight_layout(pad=0.5)
    return fig


def plot_topic_distribution(
    assigned_df: pd.DataFrame,
    plot_type: str = "Bar",
) -> go.Figure:
    """Plot distribution of documents across dominant topics.

    Args:
        assigned_df: DataFrame with 'topic_label'.
        plot_type: 'Bar' or 'Pie'.

    Returns:
        Plotly Figure.
    """
    counts = assigned_df["topic_label"].value_counts().reset_index()
    counts.columns = ["Topic", "Count"]
    counts["Percentage"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)

    if plot_type == "Pie":
        fig = px.pie(
            counts,
            names="Topic",
            values="Count",
            title="<b>Corpus Composition by Dominant Topic</b>",
            hole=0.4,
            template="plotly_white",
            hover_data=["Percentage"],
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
    else:
        fig = px.bar(
            counts,
            x="Count",
            y="Topic",
            orientation="h",
            color="Topic",
            text="Count",
            title="<b>Document Count per Discovered Topic</b>",
            template="plotly_white",
        )
        fig.update_layout(
            showlegend=False,
            xaxis_title="Number of Documents",
            yaxis_title="Topic",
            margin=dict(l=20, r=20, t=40, b=20),
            height=380,
        )
        fig.update_traces(textposition="outside")

    return fig


def plot_document_topic_heatmap(
    doc_topic_matrix: np.ndarray,
    topic_labels: List[str],
    doc_titles: List[str],
    max_docs: int = 30,
) -> go.Figure:
    """Generate a heatmap of document-topic assignment probabilities for sample documents.

    Args:
        doc_topic_matrix: (n_docs, n_topics) matrix.
        topic_labels: Names of topics for columns.
        doc_titles: Titles/IDs of documents for rows.
        max_docs: Max sample rows to display.

    Returns:
        Plotly Heatmap Figure.
    """
    n_sample = min(max_docs, doc_topic_matrix.shape[0])
    sample_matrix = np.round(doc_topic_matrix[:n_sample], 3)
    sample_titles = doc_titles[:n_sample]

    # Truncate doc titles if too long
    formatted_titles = [
        (t[:25] + "...") if len(t) > 28 else t
        for t in sample_titles
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=sample_matrix,
            x=topic_labels,
            y=formatted_titles,
            colorscale="Viridis",
            colorbar=dict(title="Weight"),
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=f"<b>Document-Topic Distribution Matrix</b> (First {n_sample} Documents)",
        xaxis_title="Topics",
        yaxis_title="Documents",
        template="plotly_white",
        height=max(400, n_sample * 22),
        margin=dict(l=20, r=20, t=40, b=40),
    )
    return fig


def plot_documents_2d_projection(
    doc_topic_matrix: np.ndarray,
    assigned_df: pd.DataFrame,
    method: str = "PCA",
    random_state: int = 42,
) -> go.Figure:
    """Project high-dimensional document-topic distributions into 2D space.

    Args:
        doc_topic_matrix: Matrix (n_docs, n_topics).
        assigned_df: DataFrame with document metadata.
        method: 'PCA' or 't-SNE'.
        random_state: Random state for reproducibility.

    Returns:
        Plotly 2D Scatter Figure.
    """
    n_samples = doc_topic_matrix.shape[0]

    if n_samples < 3:
        # Fallback empty figure if too few samples
        fig = go.Figure()
        fig.add_annotation(text="Insufficient documents for 2D projection (< 3)", showarrow=False)
        return fig

    if method.upper() == "T-SNE":
        # Perplexity must be less than n_samples
        perp = min(30, max(2, n_samples - 1))
        reducer = TSNE(
            n_components=2,
            perplexity=perp,
            random_state=random_state,
            init="pca" if n_samples > 10 else "random",
            learning_rate="auto",
        )
        coords = reducer.fit_transform(doc_topic_matrix)
        axis_names = ("t-SNE Dimension 1", "t-SNE Dimension 2")
    else:
        pca = PCA(n_components=2, random_state=random_state)
        coords = pca.fit_transform(doc_topic_matrix)
        exp_var = pca.explained_variance_ratio_ * 100
        axis_names = (
            f"PCA 1 ({exp_var[0]:.1f}% var)",
            f"PCA 2 ({exp_var[1]:.1f}% var)",
        )

    scatter_df = pd.DataFrame({
        "Dim_1": coords[:, 0],
        "Dim_2": coords[:, 1],
        "Topic": assigned_df["topic_label"].values,
        "Doc_ID": assigned_df["doc_id"].values,
        "Title": assigned_df["title"].values,
        "Confidence": assigned_df["confidence"].values,
        "Snippet": assigned_df["snippet"].values,
    })

    fig = px.scatter(
        scatter_df,
        x="Dim_1",
        y="Dim_2",
        color="Topic",
        hover_data=["Doc_ID", "Title", "Confidence", "Snippet"],
        title=f"<b>2D Document Cluster Map ({method.upper()})</b>",
        template="plotly_white",
        height=520,
    )

    fig.update_layout(
        xaxis_title=axis_names[0],
        yaxis_title=axis_names[1],
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )
    fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=1, color="DarkSlateGrey")))
    return fig


def plot_topic_search_curves(
    eval_df: pd.DataFrame,
    algorithm: str = "LDA",
) -> go.Figure:
    """Plot tuning curves for hyperparameter search over number of topics."""
    fig = go.Figure()

    if "coherence_c_v" in eval_df.columns:
        fig.add_trace(
            go.Scatter(
                x=eval_df["num_topics"],
                y=eval_df["coherence_c_v"],
                mode="lines+markers",
                name="Topic Coherence (C_v)",
                line=dict(color="#1f77b4", width=3),
                marker=dict(size=8),
                yaxis="y1",
            )
        )

    if algorithm == "LDA" and "perplexity" in eval_df.columns:
        fig.add_trace(
            go.Scatter(
                x=eval_df["num_topics"],
                y=eval_df["perplexity"],
                mode="lines+markers",
                name="Perplexity (Lower is better)",
                line=dict(color="#d62728", width=3, dash="dot"),
                marker=dict(size=8),
                yaxis="y2" if "coherence_c_v" in eval_df.columns else "y1",
            )
        )
    elif algorithm == "NMF" and "reconstruction_error" in eval_df.columns:
        fig.add_trace(
            go.Scatter(
                x=eval_df["num_topics"],
                y=eval_df["reconstruction_error"],
                mode="lines+markers",
                name="Reconstruction Error",
                line=dict(color="#ff7f0e", width=3),
                marker=dict(size=8),
                yaxis="y1",
            )
        )

    layout_args: Dict[str, Any] = dict(
        title="<b>Optimal Topic Count Search Curves</b>",
        xaxis=dict(title="Number of Topics (k)", dtick=1),
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if "coherence_c_v" in eval_df.columns and algorithm == "LDA" and "perplexity" in eval_df.columns:
        layout_args["yaxis"] = dict(title="Topic Coherence (C_v)")
        layout_args["yaxis2"] = dict(
            title="Perplexity",
            overlaying="y",
            side="right",
        )
    else:
        layout_args["yaxis"] = dict(title="Metric Value")

    fig.update_layout(**layout_args)
    return fig


def generate_pyldavis_html(
    lda_model: Any,
    dtm: Any,
    vectorizer: Any,
) -> Tuple[Optional[str], str]:
    """Generate pyLDAvis interactive HTML if package is installed."""
    try:
        import pyLDAvis
        import pyLDAvis.lda_model

        prepared_data = pyLDAvis.lda_model.prepare(
            lda_model,
            dtm,
            vectorizer,
            mds="mmds",
            sort_topics=False,
        )
        html_string = pyLDAvis.prepared_data_to_html(prepared_data)
        return html_string, "pyLDAvis visualization generated successfully."
    except ImportError:
        return None, "pyLDAvis package is not installed."
    except Exception as e:
        return None, f"pyLDAvis could not render: {str(e)}"
