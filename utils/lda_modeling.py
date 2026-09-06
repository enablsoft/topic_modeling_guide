"""LDA Topic Modeling with topic number optimization."""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Explanation of optimal topic determination
# ---------------------------------------------------------------------------
TOPIC_OPTIMIZATION_EXPLANATION = """
### How the Optimal Number of Topics Is Determined

Four complementary metrics are used. No single metric is definitive -- the best
*K* is typically where **several metrics agree**.

---

**1. Perplexity (lower is better)**
Perplexity measures how "surprised" the model is by unseen data.
Mathematically: `perplexity = exp(-log-likelihood / N)`.
A lower value means the model assigns higher probability to the held-out words.
We look for the **elbow** -- the point where perplexity stops dropping steeply
-- because continuing to add topics after that only captures noise.

**2. Log-Likelihood (higher is better)**
The total log-probability the model assigns to the training corpus.
It always increases with more topics (more parameters = better fit), so on its
own it would always favour the largest *K*. We use it alongside perplexity to
confirm the elbow, and we watch for the point where gains become marginal.

**3. UMass Coherence (closer to 0 is better)**
Coherence estimates how semantically related the top words of each topic are.
UMass coherence computes, for every pair of top words *(w_i, w_j)* in a topic:
`C = log[ D(w_i, w_j) + e ] / D(w_j)` where *D* counts documents containing
those words. Values closer to 0 indicate words that genuinely co-occur;
very negative values indicate incoherent topics.

**4. R ldatuning (4 metrics, when R is available)**
The R package `ldatuning` runs `FindTopicsNumber` which evaluates:
- **Griffiths 2004** -- log-likelihood estimated via harmonic-mean (maximize)
- **CaoJuan 2009** -- average cosine similarity of topic vectors (minimize)
- **Arun 2010** -- symmetric KL-divergence between topic and document
  distributions (minimize)
- **Deveaud 2014** -- Jensen-Shannon divergence between all topic pairs
  (maximize)

The optimal *K* is typically the value where the minimize-metrics are at
their lowest **and** the maximize-metrics are at their highest.

---

**Elbow detection** (used in this app): We draw a line from the first to the
last point of the perplexity curve, then pick the *K* whose perpendicular
distance from that line is greatest. This is a simple approximation of the
Kneedle algorithm.
"""


def get_optimization_explanation() -> str:
    """Return the explanation text for display in the app."""
    return TOPIC_OPTIMIZATION_EXPLANATION


def _check_ldatuning_available() -> bool:
    """Check if R and ldatuning package are available."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri, numpy2ri
        converter = ro.default_converter + pandas2ri.converter + numpy2ri.converter
        with converter.context():
            ro.r('library(ldatuning)')
            ro.r('library(topicmodels)')
            ro.r('library(slam)')
            ro.r('library(tm)')
        return True
    except Exception:
        return False


def find_optimal_topics(texts: List[str], min_topics: int = 2, max_topics: int = 20,
                        step: int = 1, max_features: int = 5000) -> Dict[str, Any]:
    """Find optimal number of topics using multiple metrics."""
    vectorizer = CountVectorizer(max_features=max_features, stop_words="english",
                                  min_df=2, max_df=0.95)
    dtm = vectorizer.fit_transform(texts)

    topic_range = list(range(min_topics, max_topics + 1, step))
    perplexities = []
    log_likelihoods = []
    coherence_scores = []

    progress = st.progress(0)
    status = st.empty()

    for i, n_topics in enumerate(topic_range):
        status.text(f"Evaluating {n_topics} topics...")
        lda = LatentDirichletAllocation(
            n_components=n_topics, random_state=42, max_iter=20,
            learning_method="online", n_jobs=-1
        )
        lda.fit(dtm)

        perplexities.append(lda.perplexity(dtm))
        log_likelihoods.append(lda.score(dtm))

        coherence = _compute_umass_coherence(lda, vectorizer, dtm, top_n=10)
        coherence_scores.append(coherence)

        progress.progress((i + 1) / len(topic_range))

    progress.empty()
    status.empty()

    # Find optimal using elbow method on perplexity
    optimal_idx = _find_elbow(perplexities)
    optimal_topics = topic_range[optimal_idx]

    return {
        "topic_range": topic_range,
        "perplexities": perplexities,
        "log_likelihoods": log_likelihoods,
        "coherence_scores": coherence_scores,
        "optimal_topics": optimal_topics,
    }


def find_optimal_topics_r(texts: List[str], min_topics: int = 2, max_topics: int = 20,
                           step: int = 2, max_features: int = 5000) -> Optional[Dict[str, Any]]:
    """Find optimal topics using R's ldatuning package (FindTopicsNumber).

    Returns a dict with a DataFrame of metrics per K, or None if R/ldatuning
    is not available.
    """
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri, numpy2ri

        converter = ro.default_converter + pandas2ri.converter + numpy2ri.converter
    except Exception:
        return None

    # Verify R packages are available
    try:
        with converter.context():
            ro.r('library(ldatuning)')
            ro.r('library(topicmodels)')
    except Exception:
        return None

    # Build document-term matrix in R
    vectorizer = CountVectorizer(max_features=max_features, stop_words="english",
                                  min_df=2, max_df=0.95)
    dtm_sparse = vectorizer.fit_transform(texts)
    dtm_dense = dtm_sparse.toarray()
    vocab = list(vectorizer.get_feature_names_out())

    topic_range = list(range(min_topics, max_topics + 1, step))

    with st.spinner("Running R ldatuning::FindTopicsNumber (this may take a while)..."):
        try:
            with converter.context():
                ro.globalenv["dtm_matrix"] = ro.r['matrix'](
                    ro.FloatVector(dtm_dense.flatten()),
                    nrow=dtm_dense.shape[0],
                    ncol=dtm_dense.shape[1],
                    byrow=True
                )
                ro.globalenv["vocab"] = ro.StrVector(vocab)
                ro.globalenv["topic_range"] = ro.IntVector(topic_range)

                ro.r('''
                library(slam)
                library(tm)
                library(topicmodels)
                library(ldatuning)

                # Convert dense matrix to DocumentTermMatrix
                dtm_slam <- as.simple_triplet_matrix(dtm_matrix)
                colnames(dtm_slam) <- vocab
                dtm <- as.DocumentTermMatrix(dtm_slam, weighting = weightTf)

                # Remove empty rows if any
                row_totals <- rowSums(as.matrix(dtm))
                dtm <- dtm[row_totals > 0, ]

                result <- FindTopicsNumber(
                  dtm,
                  topics = topic_range,
                  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
                  method = "Gibbs",
                  control = list(seed = 42),
                  verbose = FALSE
                )
                ''')

                # Extract column by column to avoid recarray conversion issues
                col_names = list(ro.r('colnames(result)'))
                data = {}
                for col in col_names:
                    data[col] = list(ro.r(f'result${col}'))
                results_df = pd.DataFrame(data)
            return {"results": results_df, "topic_range": topic_range}
        except Exception as e:
            st.warning(f"R ldatuning failed: {e}")
            return None


def find_optimal_k_ldatuning(ldatuning_results: Dict[str, Any]) -> int:
    """Pick the optimal K from ldatuning metrics.

    Strategy: normalise each metric to [0, 1] (respecting direction), then
    average the four scores per K and pick the K with the highest composite.
    """
    df = ldatuning_results["results"].copy()
    k_col = "topics" if "topics" in df.columns else "K"

    # Metrics and their direction (True = higher is better)
    maximize = {"Griffiths2004": True, "CaoJuan2009": False,
                "Arun2010": False, "Deveaud2014": True}

    composite = np.zeros(len(df))
    n_metrics = 0
    for metric, higher_better in maximize.items():
        if metric not in df.columns:
            continue
        vals = df[metric].values.astype(float)
        vmin, vmax = vals.min(), vals.max()
        if vmax - vmin < 1e-12:
            norm = np.ones_like(vals) * 0.5
        else:
            norm = (vals - vmin) / (vmax - vmin)
        if not higher_better:
            norm = 1.0 - norm
        composite += norm
        n_metrics += 1

    if n_metrics == 0:
        return int(df[k_col].iloc[0])

    best_idx = int(np.argmax(composite))
    return int(df[k_col].iloc[best_idx])


def _compute_umass_coherence(lda, vectorizer, dtm, top_n: int = 10) -> float:
    """Compute UMass coherence score approximation."""
    dtm_array = dtm.toarray()
    epsilon = 1e-12

    total_coherence = 0
    for topic_idx in range(lda.n_components):
        top_word_indices = lda.components_[topic_idx].argsort()[-top_n:][::-1]
        topic_coherence = 0
        pairs = 0
        for i in range(1, len(top_word_indices)):
            for j in range(i):
                w_i = top_word_indices[i]
                w_j = top_word_indices[j]
                co_doc = np.sum((dtm_array[:, w_i] > 0) & (dtm_array[:, w_j] > 0))
                d_j = np.sum(dtm_array[:, w_j] > 0)
                topic_coherence += np.log((co_doc + epsilon) / (d_j + epsilon))
                pairs += 1
        if pairs > 0:
            total_coherence += topic_coherence / pairs

    return total_coherence / lda.n_components


def _find_elbow(values: List[float]) -> int:
    """Find elbow point in a curve using the kneedle algorithm approximation."""
    if len(values) <= 2:
        return 0

    x = np.arange(len(values))
    y = np.array(values)
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    distances = []
    for i in range(len(x_norm)):
        p = np.array([x_norm[i], y_norm[i]])
        d = np.abs(np.cross(p2 - p1, p1 - p)) / (np.linalg.norm(p2 - p1) + 1e-12)
        distances.append(d)

    return int(np.argmax(distances))


def run_lda_multicore(texts: List[str], n_topics: int, max_features: int = 5000,
                      n_top_words: int = 15, iterations: int = 400,
                      passes: int = 20) -> Dict[str, Any]:
    """Run Gensim LdaMulticore and return standard results."""
    import multiprocessing
    try:
        from gensim.models import LdaMulticore
        from gensim import corpora
        from gensim.utils import simple_preprocess
    except ImportError:
        raise ImportError("gensim is required. Install with: pip install gensim")

    n_workers = max(1, multiprocessing.cpu_count() - 1)

    tokenized = [simple_preprocess(t, deacc=True) for t in texts]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=2, no_above=0.95, keep_n=max_features)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized]

    with st.spinner(f"Fitting LdaMulticore ({n_topics} topics, {passes} passes, {n_workers} workers)..."):
        model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=n_topics,
            passes=passes,
            iterations=iterations,
            workers=n_workers,
            random_state=42,
        )

    topics_matrix = model.get_topics()  # (n_topics, vocab_size)
    feature_names = [dictionary[i] for i in range(len(dictionary))]

    topics = {}
    topic_word_weights = {}
    for topic_idx in range(n_topics):
        top_indices = topics_matrix[topic_idx].argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_weights = [float(topics_matrix[topic_idx][i]) for i in top_indices]
        topics[f"Topic {topic_idx + 1}"] = top_words
        topic_word_weights[f"Topic {topic_idx + 1}"] = list(zip(top_words, top_weights))

    topic_labels = {name: ", ".join(words[:3]) for name, words in topics.items()}

    doc_topic_matrix = np.zeros((len(corpus), n_topics))
    for doc_idx, doc_bow in enumerate(corpus):
        dist = model.get_document_topics(doc_bow, minimum_probability=0.0)
        for topic_id, prob in dist:
            if topic_id < n_topics:
                doc_topic_matrix[doc_idx, topic_id] = prob

    doc_topic_df = pd.DataFrame(
        doc_topic_matrix,
        columns=[f"Topic {i+1}" for i in range(n_topics)]
    )
    doc_topic_df["dominant_topic"] = doc_topic_df.idxmax(axis=1)
    doc_topic_df["dominant_topic_prob"] = doc_topic_df.iloc[:, :n_topics].max(axis=1)
    doc_topic_df["text_preview"] = [t[:100] + "..." if len(t) > 100 else t for t in texts]

    topic_corr = np.corrcoef(doc_topic_matrix.T)
    topic_sizes = doc_topic_df["dominant_topic"].value_counts().sort_index()
    _log_perp = model.log_perplexity(corpus)
    perplexity = float(np.exp(-_log_perp))
    log_likelihood = float(_log_perp)

    return {
        "model": model,
        "model_type": "multicore",
        "gensim_dictionary": dictionary,
        "gensim_corpus": corpus,
        "vectorizer": None,
        "dtm": None,
        "topics": topics,
        "topic_word_weights": topic_word_weights,
        "topic_labels": topic_labels,
        "doc_topic_matrix": doc_topic_matrix,
        "doc_topic_df": doc_topic_df,
        "topic_correlations": topic_corr,
        "topic_sizes": topic_sizes,
        "feature_names": feature_names,
        "n_topics": n_topics,
        "perplexity": perplexity,
        "log_likelihood": log_likelihood,
    }


def run_lda(texts: List[str], n_topics: int, max_features: int = 5000,
            max_iter: int = 50, n_top_words: int = 15) -> Dict[str, Any]:
    """Run LDA topic modeling and return comprehensive results."""
    count_vectorizer = CountVectorizer(max_features=max_features, stop_words="english",
                                        min_df=2, max_df=0.95)
    dtm = count_vectorizer.fit_transform(texts)
    feature_names = count_vectorizer.get_feature_names_out()

    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english",
                                        min_df=2, max_df=0.95)
    tfidf_dtm = tfidf_vectorizer.fit_transform(texts)

    with st.spinner("Fitting LDA model..."):
        lda = LatentDirichletAllocation(
            n_components=n_topics, random_state=42, max_iter=max_iter,
            learning_method="online", n_jobs=-1
        )
        doc_topic_matrix = lda.fit_transform(dtm)

    topics = {}
    topic_word_weights = {}
    for topic_idx in range(n_topics):
        top_indices = lda.components_[topic_idx].argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_weights = [lda.components_[topic_idx][i] for i in top_indices]
        topics[f"Topic {topic_idx + 1}"] = top_words
        topic_word_weights[f"Topic {topic_idx + 1}"] = list(zip(top_words, top_weights))

    topic_labels = {}
    for topic_name, words in topics.items():
        topic_labels[topic_name] = ", ".join(words[:3])

    doc_topic_df = pd.DataFrame(
        doc_topic_matrix,
        columns=[f"Topic {i+1}" for i in range(n_topics)]
    )
    doc_topic_df["dominant_topic"] = doc_topic_df.idxmax(axis=1)
    doc_topic_df["dominant_topic_prob"] = doc_topic_df.iloc[:, :n_topics].max(axis=1)
    doc_topic_df["text_preview"] = [t[:100] + "..." if len(t) > 100 else t for t in texts]

    topic_corr = np.corrcoef(doc_topic_matrix.T)
    topic_sizes = doc_topic_df["dominant_topic"].value_counts().sort_index()
    perplexity = lda.perplexity(dtm)
    log_likelihood = lda.score(dtm)

    return {
        "model": lda,
        "vectorizer": count_vectorizer,
        "dtm": dtm,
        "topics": topics,
        "topic_word_weights": topic_word_weights,
        "topic_labels": topic_labels,
        "doc_topic_matrix": doc_topic_matrix,
        "doc_topic_df": doc_topic_df,
        "topic_correlations": topic_corr,
        "topic_sizes": topic_sizes,
        "feature_names": feature_names,
        "n_topics": n_topics,
        "perplexity": perplexity,
        "log_likelihood": log_likelihood,
    }


def plot_topic_tuning(tuning_results: Dict[str, Any]) -> go.Figure:
    """Plot topic tuning metrics."""
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Perplexity (lower=better)",
                                        "Log-Likelihood (higher=better)",
                                        "UMass Coherence (closer to 0=better)"))

    topic_range = tuning_results["topic_range"]

    fig.add_trace(
        go.Scatter(x=topic_range, y=tuning_results["perplexities"],
                   mode="lines+markers", name="Perplexity"),
        row=1, col=1
    )
    optimal = tuning_results["optimal_topics"]
    opt_idx = topic_range.index(optimal)
    fig.add_trace(
        go.Scatter(x=[optimal], y=[tuning_results["perplexities"][opt_idx]],
                   mode="markers", marker=dict(size=15, color="red", symbol="star"),
                   name=f"Optimal ({optimal})"),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=topic_range, y=tuning_results["log_likelihoods"],
                   mode="lines+markers", name="Log-Likelihood", line=dict(color="green")),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=topic_range, y=tuning_results["coherence_scores"],
                   mode="lines+markers", name="Coherence", line=dict(color="orange")),
        row=1, col=3
    )

    fig.update_layout(height=400, title_text="Topic Number Optimization", showlegend=True)
    fig.update_xaxes(title_text="Number of Topics")
    return fig


def plot_ldatuning_results(ldatuning_results: Dict[str, Any],
                           optimal_k: Optional[int] = None) -> go.Figure:
    """Plot R ldatuning FindTopicsNumber results with optional optimal K marker."""
    from plotly.subplots import make_subplots

    df = ldatuning_results["results"]
    # Expected columns: topics, Griffiths2004, CaoJuan2009, Arun2010, Deveaud2014
    metric_cols = [c for c in df.columns if c not in ("topics", "K")]
    k_col = "topics" if "topics" in df.columns else "K"
    n = len(metric_cols)

    fig = make_subplots(rows=1, cols=n, subplot_titles=metric_cols)

    # Metrics to minimize vs maximize
    minimize = {"CaoJuan2009", "Arun2010"}

    for i, metric in enumerate(metric_cols):
        color = "red" if metric in minimize else "green"
        direction = "minimize" if metric in minimize else "maximize"
        fig.add_trace(
            go.Scatter(x=df[k_col], y=df[metric], mode="lines+markers",
                       name=f"{metric} ({direction})", line=dict(color=color)),
            row=1, col=i + 1
        )

        # Mark the optimal K on each subplot
        if optimal_k is not None and optimal_k in df[k_col].values:
            opt_row = df[df[k_col] == optimal_k].iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=[optimal_k], y=[opt_row[metric]],
                    mode="markers",
                    marker=dict(size=14, color="gold", symbol="star",
                                line=dict(width=2, color="black")),
                    name=f"Optimal (K={optimal_k})",
                    showlegend=(i == 0),
                ),
                row=1, col=i + 1,
            )

    fig.update_layout(height=400, title_text="R ldatuning: FindTopicsNumber Metrics",
                      showlegend=True)
    fig.update_xaxes(title_text="K")
    return fig


def plot_topic_words(results: Dict[str, Any], n_words: int = 10) -> go.Figure:
    """Plot top words per topic as horizontal bar charts."""
    from plotly.subplots import make_subplots

    n_topics = results["n_topics"]
    cols = min(3, n_topics)
    rows = (n_topics + cols - 1) // cols

    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"Topic {i+1}" for i in range(n_topics)])

    for idx in range(n_topics):
        topic_name = f"Topic {idx + 1}"
        word_weights = results["topic_word_weights"][topic_name][:n_words]
        words = [w for w, _ in word_weights][::-1]
        weights = [w for _, w in word_weights][::-1]

        row = idx // cols + 1
        col = idx % cols + 1

        fig.add_trace(
            go.Bar(y=words, x=weights, orientation="h", name=topic_name,
                   marker_color=px.colors.qualitative.Set3[idx % len(px.colors.qualitative.Set3)]),
            row=row, col=col
        )

    fig.update_layout(height=300 * rows, showlegend=False, title_text="Top Words per Topic")
    return fig


def plot_topic_distribution(results: Dict[str, Any]) -> go.Figure:
    """Plot topic size distribution."""
    sizes = results["topic_sizes"]
    fig = px.bar(
        x=sizes.index, y=sizes.values,
        labels={"x": "Topic", "y": "Number of Documents"},
        title="Document Distribution Across Topics",
        color=sizes.index,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_topic_correlations(results: Dict[str, Any]) -> go.Figure:
    """Plot topic correlation heatmap."""
    corr = results["topic_correlations"]
    labels = [f"Topic {i+1}" for i in range(results["n_topics"])]

    fig = go.Figure(data=go.Heatmap(
        z=corr, x=labels, y=labels,
        colorscale="RdBu_r", zmid=0,
        text=np.round(corr, 2), texttemplate="%{text}",
        hovertemplate="Row: %{y}<br>Col: %{x}<br>Correlation: %{z:.3f}<extra></extra>"
    ))
    fig.update_layout(title="Topic Correlations", height=500, width=600)
    return fig


def plot_doc_topic_heatmap(results: Dict[str, Any], max_docs: int = 50) -> go.Figure:
    """Plot document-topic probability heatmap."""
    matrix = results["doc_topic_matrix"][:max_docs]
    labels = [f"Topic {i+1}" for i in range(results["n_topics"])]
    doc_labels = [f"Doc {i+1}" for i in range(matrix.shape[0])]

    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=labels, y=doc_labels,
        colorscale="YlOrRd",
        hovertemplate="Document: %{y}<br>Topic: %{x}<br>Probability: %{z:.3f}<extra></extra>"
    ))
    fig.update_layout(title=f"Document-Topic Probability Matrix (first {max_docs} docs)",
                      height=max(400, max_docs * 15), width=700)
    return fig


def plot_topic_evolution(results: Dict[str, Any], dates: pd.Series,
                         freq: str = "Y") -> Optional[go.Figure]:
    """Plot topic evolution over time if dates available."""
    if dates is None or dates.empty:
        return None

    doc_topic_df = results["doc_topic_df"].copy()
    doc_topic_df["date"] = dates.values[:len(doc_topic_df)]
    doc_topic_df["period"] = doc_topic_df["date"].dt.to_period(freq).astype(str)

    topic_cols = [f"Topic {i+1}" for i in range(results["n_topics"])]
    evolution = doc_topic_df.groupby("period")[topic_cols].mean()

    fig = go.Figure()
    for col in topic_cols:
        fig.add_trace(go.Scatter(
            x=evolution.index, y=evolution[col],
            mode="lines+markers", name=col
        ))

    fig.update_layout(
        title="Topic Evolution Over Time",
        xaxis_title="Time Period",
        yaxis_title="Average Topic Proportion",
        height=500
    )
    return fig


def generate_pyldavis(results: Dict[str, Any]) -> Optional[str]:
    """Generate pyLDAvis HTML visualization."""
    try:
        import pyLDAvis
        if results.get("model_type") == "multicore":
            import pyLDAvis.gensim_models
            vis_data = pyLDAvis.gensim_models.prepare(
                results["model"],
                results["gensim_corpus"],
                results["gensim_dictionary"],
                mds="mmds",
                sort_topics=False,
            )
        else:
            import pyLDAvis.lda_model
            vis_data = pyLDAvis.lda_model.prepare(
                results["model"], results["dtm"], results["vectorizer"],
                mds="mmds", sort_topics=False,
            )
        return pyLDAvis.prepared_data_to_html(vis_data)
    except Exception as e:
        st.warning(f"pyLDAvis visualization failed: {e}")
        return None


def plot_word_clouds(results: Dict[str, Any]) -> List:
    """Generate word clouds for each topic."""
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    figures = []
    n_topics = results["n_topics"]

    for idx in range(n_topics):
        topic_name = f"Topic {idx + 1}"
        word_weights = dict(results["topic_word_weights"][topic_name])

        wc = WordCloud(width=400, height=300, background_color="white",
                       colormap="viridis", max_words=30)
        wc.generate_from_frequencies(word_weights)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"Topic {idx + 1}: {results['topic_labels'][topic_name]}")
        ax.axis("off")
        figures.append(fig)
        plt.close(fig)

    return figures
