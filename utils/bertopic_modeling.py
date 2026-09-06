"""BERTopic modeling with comprehensive analysis."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional


def run_bertopic(texts: List[str], embeddings: Optional[np.ndarray] = None,
                 embedding_model=None, n_topics: Optional[int] = None,
                 min_topic_size: int = 10, n_gram_range: tuple = (1, 2),
                 nr_topics: Optional[int] = None,
                 auto_topics: bool = True) -> Dict[str, Any]:
    """Run BERTopic and return comprehensive results."""
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    from bertopic.representation import KeyBERTInspired

    vectorizer = CountVectorizer(stop_words="english", ngram_range=n_gram_range,
                                  min_df=2, max_df=0.95)

    # KeyBERTInspired re-embeds candidate words with BERTopic's internal model and
    # computes cosine similarity against the document embeddings. When external
    # embeddings (e.g. OpenAI 1536-dim) are provided but the internal model is a
    # different SBERT variant (384-dim), this raises a dimension mismatch. Skip
    # KeyBERTInspired whenever the embedding dims could be mismatched.
    use_keybert = not (embeddings is not None and embedding_model is None)
    representation_model = KeyBERTInspired() if use_keybert else None

    # Configure BERTopic
    topic_model = BERTopic(
        embedding_model=embedding_model if embedding_model else "all-MiniLM-L6-v2",
        vectorizer_model=vectorizer,
        representation_model=representation_model,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics if (nr_topics and nr_topics > 0 and not auto_topics) else None,
        verbose=False,
        calculate_probabilities=True,
    )

    with st.spinner("Fitting BERTopic model..."):
        if embeddings is not None:
            topics, probs = topic_model.fit_transform(texts, embeddings=embeddings)
        else:
            topics, probs = topic_model.fit_transform(texts)

    n_found = len(set(topics)) - (1 if -1 in topics else 0)
    if auto_topics:
        st.info(f"Auto-detected **{n_found}** topics")

    # Get topic info
    topic_info = topic_model.get_topic_info()

    # Document-topic info
    doc_info = topic_model.get_document_info(texts)

    # Topic representations
    all_topics = {}
    for topic_id in topic_model.get_topics():
        if topic_id != -1:
            words = topic_model.get_topic(topic_id)
            all_topics[f"Topic {topic_id}"] = [(w, float(s)) for w, s in words[:15]]

    # Topic word weights for cross-model comparison (same format as other models)
    topic_words = {}
    topic_word_weights = {}
    for topic_id in topic_model.get_topics():
        if topic_id != -1:
            words = topic_model.get_topic(topic_id)
            topic_words[f"Topic {topic_id}"] = [w for w, _ in words[:15]]
            topic_word_weights[f"Topic {topic_id}"] = [(w, float(s)) for w, s in words[:15]]

    # Topic sizes from topic_info
    topic_sizes_data = topic_info[topic_info["Topic"] != -1].set_index("Topic")["Count"]
    topic_sizes = topic_sizes_data.rename(index=lambda x: f"Topic {x}")

    # Document-topic probability matrix
    if probs is not None and len(probs.shape) > 1:
        n_topics_actual = probs.shape[1]
        doc_topic_df = pd.DataFrame(
            probs,
            columns=[f"Topic {i}" for i in range(n_topics_actual)]
        )
    else:
        doc_topic_df = pd.DataFrame({"topic": topics})

    doc_topic_df["text_preview"] = [t[:100] + "..." if len(t) > 100 else t for t in texts]
    doc_topic_df["assigned_topic"] = topics

    # Topic correlations from probabilities
    topic_corr = None
    if probs is not None and len(probs.shape) > 1:
        topic_corr = np.corrcoef(probs.T)

    return {
        "model": topic_model,
        "topics": topics,
        "probs": probs,
        "topic_info": topic_info,
        "doc_info": doc_info,
        "all_topics": all_topics,
        "topic_words": topic_words,
        "topic_word_weights": topic_word_weights,
        "topic_sizes": topic_sizes,
        "doc_topic_df": doc_topic_df,
        "topic_correlations": topic_corr,
        "n_topics": n_found,
        "texts": texts,
        "embeddings": embeddings,
    }


def label_topics_with_llm(results: Dict[str, Any],
                           openai_api_key: str,
                           model_name: str = "gpt-4o-mini") -> Dict[str, Any]:
    """Use OpenAI to generate topic labels via BERTopic's OpenAI representation."""
    try:
        import openai
        from bertopic.representation import OpenAI as BertopicOpenAI

        client = openai.OpenAI(api_key=openai_api_key)
        repr_model = BertopicOpenAI(
            client,
            model=model_name,
            nr_docs=4,
        )
        topic_model = results["model"]
        topic_model.update_topics(results["texts"], representation_model=repr_model)

        # Extract LLM-generated labels
        llm_names = []
        llm_descriptions = []
        for topic_id in sorted(topic_model.get_topics().keys()):
            if topic_id == -1:
                continue
            words = topic_model.get_topic(topic_id)
            label = ", ".join(w for w, _ in words[:3])
            llm_names.append(label)
            llm_descriptions.append(", ".join(w for w, _ in words[:8]))

        results["llm_topic_names"] = llm_names
        results["llm_topic_descriptions"] = llm_descriptions

        # Update topic_info
        results["topic_info"] = topic_model.get_topic_info()
    except Exception as e:
        st.warning(f"BERTopic LLM labeling failed: {e}")

    return results


def plot_topics_barchart(results: Dict[str, Any]) -> Optional[Any]:
    """Plot BERTopic barchart visualization."""
    try:
        fig = results["model"].visualize_barchart(top_n_topics=12, n_words=10, height=400)
        return fig
    except Exception as e:
        st.warning(f"Barchart visualization failed: {e}")
        return None


def plot_topics_scatter(results: Dict[str, Any]) -> Optional[Any]:
    """Plot BERTopic topic scatter/intertopic distance map."""
    try:
        fig = results["model"].visualize_topics()
        return fig
    except Exception as e:
        st.warning(f"Topic scatter visualization failed: {e}")
        return None


def plot_hierarchy(results: Dict[str, Any]) -> Optional[Any]:
    """Plot BERTopic topic hierarchy."""
    try:
        fig = results["model"].visualize_hierarchy()
        return fig
    except Exception as e:
        st.warning(f"Hierarchy visualization failed: {e}")
        return None


def plot_heatmap(results: Dict[str, Any]) -> Optional[Any]:
    """Plot BERTopic topic similarity heatmap."""
    try:
        fig = results["model"].visualize_heatmap()
        return fig
    except Exception as e:
        st.warning(f"Heatmap visualization failed: {e}")
        return None


def plot_documents(results: Dict[str, Any]) -> Optional[Any]:
    """Plot BERTopic document visualization."""
    try:
        fig = results["model"].visualize_documents(
            results["texts"],
            embeddings=results["embeddings"],
            hide_annotations=True
        )
        return fig
    except Exception as e:
        st.warning(f"Document visualization failed: {e}")
        return None


def plot_topic_evolution(results: Dict[str, Any], timestamps: pd.Series,
                         nr_bins: int = 10) -> Optional[Any]:
    """Plot BERTopic topics over time."""
    try:
        topics_over_time = results["model"].topics_over_time(
            results["texts"],
            timestamps.tolist(),
            nr_bins=nr_bins
        )
        fig = results["model"].visualize_topics_over_time(topics_over_time)
        return fig, topics_over_time
    except Exception as e:
        st.warning(f"Topic evolution visualization failed: {e}")
        return None, None


def plot_topic_correlations(results: Dict[str, Any]) -> Optional[go.Figure]:
    """Plot topic correlation heatmap from probabilities."""
    corr = results.get("topic_correlations")
    if corr is None:
        return None

    n = corr.shape[0]
    labels = [f"Topic {i}" for i in range(n)]

    fig = go.Figure(data=go.Heatmap(
        z=corr, x=labels, y=labels,
        colorscale="RdBu_r", zmid=0,
        text=np.round(corr, 2), texttemplate="%{text}",
    ))
    fig.update_layout(title="BERTopic: Topic Correlations", height=500)
    return fig


def get_representative_docs(results: Dict[str, Any], topic_id: int, n: int = 5) -> List[str]:
    """Get representative documents for a topic."""
    try:
        docs = results["model"].get_representative_docs(topic_id)
        return docs[:n]
    except Exception:
        return []


def reduce_topics(results: Dict[str, Any], nr_topics: int) -> Dict[str, Any]:
    """Reduce the number of topics."""
    model = results["model"]
    model.reduce_topics(results["texts"], nr_topics=nr_topics)

    # Update results
    results["topics"] = model.topics_
    results["topic_info"] = model.get_topic_info()
    results["doc_info"] = model.get_document_info(results["texts"])

    return results


def get_topic_tree(results: Dict[str, Any]) -> Optional[str]:
    """Get topic tree as text."""
    try:
        hierarchical_topics = results["model"].hierarchical_topics(results["texts"])
        tree = results["model"].get_topic_tree(hierarchical_topics)
        return tree
    except Exception as e:
        st.warning(f"Topic tree generation failed: {e}")
        return None
