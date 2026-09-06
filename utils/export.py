"""Export utilities for results and Jupyter notebook generation."""

import io
import json
import zipfile
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


def generate_notebook(config: Dict[str, Any], results: Dict[str, Dict[str, Any]]) -> str:
    """Generate a Jupyter notebook with all the analysis code."""
    import nbformat
    from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

    nb = new_notebook()
    cells = []

    # Title
    cells.append(new_markdown_cell("# Topic Modeling Analysis\n\nAuto-generated notebook."))

    # Imports
    cells.append(new_code_cell("""import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
"""))

    # Data loading
    data_source = config.get("data_source", "csv")
    if data_source == "csv":
        cells.append(new_markdown_cell("## Data Loading"))
        cells.append(new_code_cell(f"""# Load your CSV file
df = pd.read_csv("your_data.csv")
text_columns = {config.get('text_columns', ['text'])}
texts = df[text_columns].astype(str).agg(' '.join, axis=1).tolist()
print(f"Loaded {{len(texts)}} documents")
df.head()
"""))
    else:
        cells.append(new_markdown_cell("## Data Loading"))
        cells.append(new_code_cell("""import os
folder_path = "your_corpus_folder"
texts = []
for fname in sorted(os.listdir(folder_path)):
    if fname.endswith('.txt'):
        with open(os.path.join(folder_path, fname), 'r') as f:
            texts.append(f.read().strip())
print(f"Loaded {len(texts)} documents")
"""))

    # Embedding model
    embedding_model = config.get("embedding_model", "SBERT (all-MiniLM-L6-v2)")
    cells.append(new_markdown_cell("## Embedding Model"))

    if "SBERT" in embedding_model:
        model_id = "all-MiniLM-L6-v2"
        if "mpnet" in embedding_model:
            model_id = "all-mpnet-base-v2"
        cells.append(new_code_cell(f"""from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("{model_id}")
embeddings = embedding_model.encode(texts, show_progress_bar=True, batch_size=64)
print(f"Embeddings shape: {{embeddings.shape}}")
"""))
    elif "OpenAI" in embedding_model:
        model_id = "text-embedding-3-small" if "small" in embedding_model else "text-embedding-3-large"
        cells.append(new_code_cell(f"""import openai

client = openai.OpenAI(api_key="YOUR_API_KEY")
embeddings = []
batch_size = 100
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    response = client.embeddings.create(input=batch, model="{model_id}")
    embeddings.extend([item.embedding for item in response.data])
embeddings = np.array(embeddings)
print(f"Embeddings shape: {{embeddings.shape}}")
"""))
    elif "Voyage" in embedding_model:
        model_id = "voyage-3" if "lite" not in embedding_model else "voyage-3-lite"
        cells.append(new_code_cell(f"""import voyageai

client = voyageai.Client(api_key="YOUR_API_KEY")
embeddings = []
batch_size = 100
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    result = client.embed(batch, model="{model_id}", input_type="document")
    embeddings.extend(result.embeddings)
embeddings = np.array(embeddings)
print(f"Embeddings shape: {{embeddings.shape}}")
"""))
    elif "Gemini" in embedding_model:
        cells.append(new_code_cell("""from google import genai

client = genai.Client(api_key="YOUR_API_KEY")
embeddings = []
batch_size = 100
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    result = client.models.embed_content(model="gemini-embedding-001", contents=batch)
    for emb in result.embeddings:
        embeddings.append(emb.values)
embeddings = np.array(embeddings)
print(f"Embeddings shape: {embeddings.shape}")
"""))

    # LDA section
    if "lda" in results:
        cells.append(new_markdown_cell("## LDA Topic Modeling"))
        n_topics = results["lda"].get("n_topics", 10)
        cells.append(new_code_cell(f"""from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Create document-term matrix
vectorizer = CountVectorizer(max_features=5000, stop_words='english', min_df=2, max_df=0.95)
dtm = vectorizer.fit_transform(texts)
feature_names = vectorizer.get_feature_names_out()

# Topic number optimization
perplexities = []
log_likelihoods = []
topic_range = range(2, 21)
for k in topic_range:
    lda_temp = LatentDirichletAllocation(n_components=k, random_state=42,
                                          max_iter=20, learning_method='online')
    lda_temp.fit(dtm)
    perplexities.append(lda_temp.perplexity(dtm))
    log_likelihoods.append(lda_temp.score(dtm))

fig = make_subplots(rows=1, cols=2, subplot_titles=('Perplexity', 'Log-Likelihood'))
fig.add_trace(go.Scatter(x=list(topic_range), y=perplexities, mode='lines+markers'), row=1, col=1)
fig.add_trace(go.Scatter(x=list(topic_range), y=log_likelihoods, mode='lines+markers'), row=1, col=2)
fig.update_layout(title='LDA Topic Number Optimization')
fig.show()
"""))

        cells.append(new_code_cell(f"""# Fit final LDA model
n_topics = {n_topics}
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42,
                                 max_iter=50, learning_method='online')
doc_topic_matrix = lda.fit_transform(dtm)

# Extract topics
for i in range(n_topics):
    top_indices = lda.components_[i].argsort()[-15:][::-1]
    top_words = [feature_names[idx] for idx in top_indices]
    print(f"Topic {{i+1}}: {{', '.join(top_words)}}")
"""))

        cells.append(new_code_cell(f"""# Document-topic distribution
doc_topic_df = pd.DataFrame(doc_topic_matrix,
                             columns=[f'Topic {{i+1}}' for i in range({n_topics})])
doc_topic_df['dominant_topic'] = doc_topic_df.idxmax(axis=1)
doc_topic_df['dominant_topic_prob'] = doc_topic_df.iloc[:, :{n_topics}].max(axis=1)
doc_topic_df.head(10)
"""))

        cells.append(new_code_cell(f"""# Topic correlations
topic_corr = np.corrcoef(doc_topic_matrix.T)
labels = [f'Topic {{i+1}}' for i in range({n_topics})]
fig = go.Figure(data=go.Heatmap(z=topic_corr, x=labels, y=labels,
                                 colorscale='RdBu_r', zmid=0))
fig.update_layout(title='LDA Topic Correlations')
fig.show()
"""))

        cells.append(new_code_cell("""# pyLDAvis interactive visualization
try:
    import pyLDAvis
    import pyLDAvis.lda_model
    vis_data = pyLDAvis.lda_model.prepare(lda, dtm, vectorizer, mds='mmds')
    pyLDAvis.display(vis_data)
except ImportError:
    print("Install pyLDAvis: pip install pyLDAvis")
"""))

        cells.append(new_code_cell("""# Word clouds
from wordcloud import WordCloud
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, (n_topics + 1) // 2, figsize=(20, 10))
for i, ax in enumerate(axes.flat):
    if i >= n_topics:
        ax.set_visible(False)
        continue
    top_indices = lda.components_[i].argsort()[-30:][::-1]
    word_weights = {feature_names[idx]: lda.components_[i][idx] for idx in top_indices}
    wc = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(word_weights)
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(f'Topic {i+1}')
    ax.axis('off')
plt.tight_layout()
plt.show()
"""))

    # BERTopic section
    if "bertopic" in results:
        cells.append(new_markdown_cell("## BERTopic"))
        cells.append(new_code_cell("""from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired

vectorizer_model = CountVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2, max_df=0.95)
representation_model = KeyBERTInspired()

topic_model = BERTopic(
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    min_topic_size=10,
    calculate_probabilities=True,
    verbose=False,
)
topics, probs = topic_model.fit_transform(texts, embeddings=embeddings)
print(f"Found {len(set(topics)) - (1 if -1 in topics else 0)} topics")
topic_model.get_topic_info().head(15)
"""))

        cells.append(new_code_cell("""# BERTopic visualizations
fig = topic_model.visualize_barchart(top_n_topics=12, n_words=10)
fig.show()
"""))

        cells.append(new_code_cell("""# Intertopic distance map
fig = topic_model.visualize_topics()
fig.show()
"""))

        cells.append(new_code_cell("""# Topic hierarchy
fig = topic_model.visualize_hierarchy()
fig.show()
"""))

        cells.append(new_code_cell("""# Topic similarity heatmap
fig = topic_model.visualize_heatmap()
fig.show()
"""))

        cells.append(new_code_cell("""# Document-topic visualization
fig = topic_model.visualize_documents(texts, embeddings=embeddings, hide_annotations=True)
fig.show()
"""))

        cells.append(new_code_cell("""# Document info with topics
doc_info = topic_model.get_document_info(texts)
doc_info.head(10)
"""))

    # Turftopic section
    if "turftopic" in results:
        model_type = results["turftopic"].get("model_type", "KeyNMF")
        n_topics_tt = results["turftopic"].get("n_topics", 10)
        cells.append(new_markdown_cell(f"## Turftopic ({model_type})"))
        cells.append(new_code_cell(f"""import turftopic

model = turftopic.{model_type}(n_components={n_topics_tt}, encoder="all-MiniLM-L6-v2")
doc_topic_matrix = model.fit_transform(texts)
print(f"Document-topic matrix shape: {{doc_topic_matrix.shape}}")

# Display topics
try:
    vocab = model.get_vocab()
    components = model.components_
    for i in range(min({n_topics_tt}, components.shape[0])):
        top_indices = components[i].argsort()[-10:][::-1]
        words = [vocab[idx] for idx in top_indices]
        print(f"Topic {{i+1}}: {{', '.join(words)}}")
except Exception as e:
    print(f"Could not extract topic words: {{e}}")
"""))

        cells.append(new_code_cell(f"""# Document-topic distribution
doc_topic_df = pd.DataFrame(doc_topic_matrix,
                             columns=[f'Topic {{i+1}}' for i in range(doc_topic_matrix.shape[1])])
doc_topic_df['dominant_topic'] = doc_topic_df.idxmax(axis=1)
print("Topic distribution:")
print(doc_topic_df['dominant_topic'].value_counts().sort_index())
"""))

    # BunkaTopics section
    if "bunkatopics" in results:
        n_topics_bk = results["bunkatopics"].get("n_topics", 10)
        cells.append(new_markdown_cell("## BunkaTopics"))
        cells.append(new_code_cell(f"""from bunkatopics import Bunka
from sentence_transformers import SentenceTransformer

emb_model = SentenceTransformer("all-MiniLM-L6-v2")
bunka = Bunka(embedding_model=emb_model)
bunka.fit(texts)
topics_df = bunka.get_topics(n_clusters={n_topics_bk}, name_length=5)
print(topics_df)
"""))

        cells.append(new_code_cell("""# BunkaTopics map visualization
fig = bunka.visualize_topics()
fig.show()
"""))

    # STM section
    if "stm" in results:
        n_topics_stm = results["stm"].get("n_topics", 10)
        formula = results["stm"].get("prevalence_formula", "")
        cells.append(new_markdown_cell("## Structural Topic Modeling"))
        cells.append(new_code_cell(f"""# STM requires R. Install: rpy2, and R packages: stm
# If R is not available, use the NMF approximation below

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.packages import importr
    pandas2ri.activate()
    numpy2ri.activate()
    stm = importr('stm')

    text_vector = ro.StrVector(texts)
    processed = ro.r('''
    function(texts) {{
        processed <- textProcessor(texts, metadata=NULL, lowercase=TRUE,
                                    removestopwords=TRUE, removenumbers=TRUE)
        out <- prepDocuments(processed$documents, processed$vocab, lower.thresh=2)
        return(out)
    }}
    ''')(text_vector)

    ro.globalenv['processed'] = processed
    model = ro.r('''
    stm(documents=processed$documents, vocab=processed$vocab,
        K={n_topics_stm}, max.em.its=75, init.type="Spectral", verbose=FALSE)
    ''')

    ro.globalenv['stm_model'] = model
    theta = np.array(ro.r('stm_model$theta'))
    print(f"Document-topic matrix shape: {{theta.shape}}")

    # Topic correlations
    topic_corr = np.array(ro.r('topicCorr(stm_model)$cor'))

    # Semantic coherence and exclusivity
    coherence = list(ro.r('semanticCoherence(stm_model, processed$documents)'))
    exclusivity = list(ro.r('exclusivity(stm_model)'))

except ImportError:
    print("rpy2 not available. Using NMF fallback.")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF

    tfidf = TfidfVectorizer(max_features=5000, stop_words='english', min_df=2, max_df=0.95)
    tfidf_matrix = tfidf.fit_transform(texts)
    nmf = NMF(n_components={n_topics_stm}, random_state=42, max_iter=200)
    theta = nmf.fit_transform(tfidf_matrix)
    theta = theta / (theta.sum(axis=1, keepdims=True) + 1e-12)
    print(f"Document-topic matrix shape: {{theta.shape}}")
"""))

    # Export section
    cells.append(new_markdown_cell("## Export Results"))
    cells.append(new_code_cell("""# Save document-topic matrices
# doc_topic_df.to_csv('document_topic_matrix.csv', index=False)
print("Export your results by uncommenting the lines above")
"""))

    nb.cells = cells
    return nbformat.writes(nb)


def _fig_to_html(fig) -> str:
    """Convert a Plotly figure to a self-contained HTML string."""
    return fig.to_html(full_html=True, include_plotlyjs="cdn")


def _matplotlib_fig_to_png(fig) -> bytes:
    """Convert a matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return buf.getvalue()


def _collect_lda_figures(model_results: Dict[str, Any]) -> List[tuple]:
    """Return list of (zip_path, bytes) for LDA visualizations."""
    from utils.lda_modeling import (
        plot_topic_words, plot_topic_distribution, plot_topic_correlations,
        plot_doc_topic_heatmap, plot_word_clouds, plot_topic_tuning,
        plot_ldatuning_results, find_optimal_k_ldatuning,
    )
    from utils.visualization import plot_topic_similarity_network

    figs = []

    def _add_plotly(name, fn, *args, **kwargs):
        try:
            fig = fn(*args, **kwargs)
            if fig is not None:
                figs.append((f"lda/figures/{name}.html", _fig_to_html(fig).encode()))
        except Exception:
            pass

    _add_plotly("topic_words", plot_topic_words, model_results)
    _add_plotly("topic_distribution", plot_topic_distribution, model_results)
    _add_plotly("topic_correlations", plot_topic_correlations, model_results)
    _add_plotly("doc_topic_heatmap", plot_doc_topic_heatmap, model_results, 50)

    if model_results.get("topic_correlations") is not None:
        _add_plotly("topic_network", plot_topic_similarity_network,
                    model_results["topic_correlations"], title="LDA Topic Network")

    if model_results.get("tuning") is not None:
        _add_plotly("topic_tuning", plot_topic_tuning, model_results["tuning"])

    if model_results.get("ldatuning_results") is not None:
        try:
            ld = model_results["ldatuning_results"]
            optimal_k = find_optimal_k_ldatuning(ld)
            fig = plot_ldatuning_results(ld, optimal_k=optimal_k)
            if fig is not None:
                figs.append(("lda/figures/ldatuning_results.html", _fig_to_html(fig).encode()))
        except Exception:
            pass

    # Word clouds → PNG
    try:
        wc_figs = plot_word_clouds(model_results)
        for i, wc_fig in enumerate(wc_figs):
            try:
                figs.append((f"lda/figures/wordcloud_topic_{i+1}.png", _matplotlib_fig_to_png(wc_fig)))
                import matplotlib.pyplot as plt
                plt.close(wc_fig)
            except Exception:
                pass
    except Exception:
        pass

    return figs


def _collect_bertopic_figures(model_results: Dict[str, Any]) -> List[tuple]:
    """Return list of (zip_path, bytes) for BERTopic visualizations."""
    from utils.bertopic_modeling import (
        plot_topics_barchart, plot_topics_scatter, plot_hierarchy, plot_heatmap,
        plot_documents, plot_topic_correlations as bertopic_corr,
    )

    figs = []

    def _add(name, fn, *args, **kwargs):
        try:
            fig = fn(*args, **kwargs)
            if fig is not None:
                figs.append((f"bertopic/figures/{name}.html", _fig_to_html(fig).encode()))
        except Exception:
            pass

    _add("barchart", plot_topics_barchart, model_results)
    _add("intertopic_map", plot_topics_scatter, model_results)
    _add("hierarchy", plot_hierarchy, model_results)
    _add("heatmap", plot_heatmap, model_results)
    _add("documents", plot_documents, model_results)
    _add("topic_correlations", bertopic_corr, model_results)

    return figs


def _collect_turftopic_figures(model_results: Dict[str, Any]) -> List[tuple]:
    """Return list of (zip_path, bytes) for Turftopic visualizations."""
    from utils.turftopic_modeling import (
        plot_topic_words, plot_topic_distribution,
        plot_topic_correlations, plot_doc_topic_heatmap,
    )

    figs = []

    def _add(name, fn, *args, **kwargs):
        try:
            fig = fn(*args, **kwargs)
            if fig is not None:
                figs.append((f"turftopic/figures/{name}.html", _fig_to_html(fig).encode()))
        except Exception:
            pass

    _add("topic_words", plot_topic_words, model_results)
    _add("topic_distribution", plot_topic_distribution, model_results)
    _add("topic_correlations", plot_topic_correlations, model_results)
    _add("doc_topic_heatmap", plot_doc_topic_heatmap, model_results, 50)

    return figs


def _collect_bunka_figures(model_results: Dict[str, Any]) -> List[tuple]:
    """Return list of (zip_path, bytes) for BunkaTopics visualizations."""
    from utils.bunka_modeling import (
        plot_bunka_map, plot_bunka_docs,
        plot_topic_words, plot_topic_distribution,
    )

    figs = []

    def _add(name, fn, *args, **kwargs):
        try:
            fig = fn(*args, **kwargs)
            if fig is not None:
                figs.append((f"bunkatopics/figures/{name}.html", _fig_to_html(fig).encode()))
        except Exception:
            pass

    _add("topic_map", plot_bunka_map, model_results)
    _add("document_map", plot_bunka_docs, model_results)
    _add("topic_words", plot_topic_words, model_results)
    _add("topic_distribution", plot_topic_distribution, model_results)

    return figs


def _collect_stm_figures(model_results: Dict[str, Any]) -> List[tuple]:
    """Return list of (zip_path, bytes) for STM visualizations."""
    from utils.stm_modeling import (
        plot_topic_words, plot_topic_correlations,
        plot_quality, plot_topic_distribution,
    )
    from utils.visualization import plot_topic_similarity_network

    figs = []

    def _add(name, fn, *args, **kwargs):
        try:
            fig = fn(*args, **kwargs)
            if fig is not None:
                figs.append((f"stm/figures/{name}.html", _fig_to_html(fig).encode()))
        except Exception:
            pass

    _add("topic_words", plot_topic_words, model_results)
    _add("topic_distribution", plot_topic_distribution, model_results)
    _add("topic_correlations", plot_topic_correlations, model_results)

    if model_results.get("coherence") and any(c != 0 for c in model_results["coherence"]):
        _add("topic_quality", plot_quality, model_results)

    if model_results.get("topic_correlations") is not None:
        _add("topic_network", plot_topic_similarity_network,
             model_results["topic_correlations"], title="STM Topic Network")

    return figs


_FIGURE_COLLECTORS = {
    "lda": _collect_lda_figures,
    "bertopic": _collect_bertopic_figures,
    "turftopic": _collect_turftopic_figures,
    "bunkatopics": _collect_bunka_figures,
    "stm": _collect_stm_figures,
}


def create_download_zip(results: Dict[str, Dict[str, Any]],
                         config: Dict[str, Any],
                         notebook_str: str) -> bytes:
    """Create a ZIP file with all results for download."""
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add notebook
        zf.writestr("topic_modeling_analysis.ipynb", notebook_str)

        # Add results for each model
        for model_name, model_results in results.items():
            prefix = f"{model_name}/"

            # Document-topic matrix
            if "doc_topic_df" in model_results:
                csv_buffer = io.StringIO()
                model_results["doc_topic_df"].to_csv(csv_buffer, index=False)
                zf.writestr(f"{prefix}document_topic_matrix.csv", csv_buffer.getvalue())

            # Topics summary — use topic_word_weights (always a dict) because
            # BERTopic stores per-doc topic IDs in "topics" (a list, not a dict).
            if "topic_word_weights" in model_results and isinstance(model_results["topic_word_weights"], dict):
                topics_df = pd.DataFrame([
                    {"Topic": name, "Top Words": ", ".join(w for w, _ in ww[:15])}
                    for name, ww in model_results["topic_word_weights"].items()
                ])
                csv_buffer = io.StringIO()
                topics_df.to_csv(csv_buffer, index=False)
                zf.writestr(f"{prefix}topics_summary.csv", csv_buffer.getvalue())
            elif "topics" in model_results and isinstance(model_results["topics"], dict):
                topics_df = pd.DataFrame([
                    {"Topic": k, "Top Words": ", ".join(v)}
                    for k, v in model_results["topics"].items()
                ])
                csv_buffer = io.StringIO()
                topics_df.to_csv(csv_buffer, index=False)
                zf.writestr(f"{prefix}topics_summary.csv", csv_buffer.getvalue())

            # Topic correlations
            if "topic_correlations" in model_results and model_results["topic_correlations"] is not None:
                corr = model_results["topic_correlations"]
                n = corr.shape[0]
                labels = [f"Topic {i+1}" for i in range(n)]
                corr_df = pd.DataFrame(corr, index=labels, columns=labels)
                csv_buffer = io.StringIO()
                corr_df.to_csv(csv_buffer)
                zf.writestr(f"{prefix}topic_correlations.csv", csv_buffer.getvalue())

            # Topic info for BERTopic
            if "topic_info" in model_results:
                csv_buffer = io.StringIO()
                model_results["topic_info"].to_csv(csv_buffer, index=False)
                zf.writestr(f"{prefix}topic_info.csv", csv_buffer.getvalue())

            # Doc info for BERTopic
            if "doc_info" in model_results:
                csv_buffer = io.StringIO()
                model_results["doc_info"].to_csv(csv_buffer, index=False)
                zf.writestr(f"{prefix}document_info.csv", csv_buffer.getvalue())

            # Topics DF for BunkaTopics
            if "topics_df" in model_results and model_results["topics_df"] is not None:
                csv_buffer = io.StringIO()
                model_results["topics_df"].to_csv(csv_buffer, index=False)
                zf.writestr(f"{prefix}bunka_topics.csv", csv_buffer.getvalue())

            # STM quality metrics
            if "coherence" in model_results and any(c != 0 for c in model_results["coherence"]):
                quality_df = pd.DataFrame({
                    "Topic": [f"Topic {i+1}" for i in range(model_results["n_topics"])],
                    "Semantic Coherence": model_results["coherence"],
                    "Exclusivity": model_results["exclusivity"],
                })
                csv_buffer = io.StringIO()
                quality_df.to_csv(csv_buffer, index=False)
                zf.writestr(f"{prefix}topic_quality.csv", csv_buffer.getvalue())

            # Visualizations — HTML (Plotly) and PNG (matplotlib)
            collector = _FIGURE_COLLECTORS.get(model_name)
            if collector is not None:
                try:
                    for zip_path, fig_bytes in collector(model_results):
                        zf.writestr(zip_path, fig_bytes)
                except Exception:
                    pass

        # Add config info
        config_info = {k: str(v) for k, v in config.items()}
        zf.writestr("analysis_config.json", json.dumps(config_info, indent=2))

    buffer.seek(0)
    return buffer.getvalue()
