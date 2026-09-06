"""
Topic Modeling Streamlit Application
=====================================
Supports LDA, BERTopic, Turftopic, BunkaTopics, and Structural Topic Modeling.
Embedding options: SBERT, OpenAI, Voyage, Gemini.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from utils.data_loader import (
    load_csv, load_folder, concatenate_columns, preprocess_texts,
    get_text_columns, get_metadata_columns, detect_time_column,
    chunk_texts, aggregate_chunk_results, advanced_preprocess,
)
from utils.sample_data import get_sample_datasets, load_sample_dataset
from utils.embeddings import (
    get_available_models, requires_api_key, get_api_key_label, compute_embeddings
)
from utils.export import generate_notebook, create_download_zip

# ---------------------------------------------------------------------------
# Cached preprocessing helpers (avoid re-computation on every widget change)
# ---------------------------------------------------------------------------
@st.cache_data
def _cached_get_texts(df, selected_cols_tuple, separator):
    texts_series = concatenate_columns(df, list(selected_cols_tuple), separator)
    return preprocess_texts(texts_series)

@st.cache_data
def _cached_advanced_preprocess(texts_tuple, lowercase, remove_punctuation,
                                  remove_numbers, remove_stopwords, lemmatize,
                                  content_words_only, custom_stopwords_tuple):
    opts = dict(
        lowercase=lowercase, remove_punctuation=remove_punctuation,
        remove_numbers=remove_numbers, remove_stopwords=remove_stopwords,
        lemmatize=lemmatize, content_words_only=content_words_only,
    )
    if custom_stopwords_tuple:
        opts["custom_stopwords"] = list(custom_stopwords_tuple)
    return advanced_preprocess(list(texts_tuple), **opts)

@st.cache_data
def _cached_chunk_texts(texts_tuple, max_words, overlap):
    return chunk_texts(list(texts_tuple), max_words, overlap)

def _verify_llm_connection(provider: str, api_key: str, model_name: str) -> tuple:
    """Send a minimal request to the chosen LLM provider and return (ok, message)."""
    if not api_key:
        return False, "No API key provided."
    try:
        if provider == "OpenAI":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": "ping"}], max_tokens=1,
            )
            return True, f"Connected to OpenAI ({model_name})."
        elif provider == "Anthropic (Claude)":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            client.messages.create(
                model=model_name, max_tokens=1, messages=[{"role": "user", "content": "ping"}],
            )
            return True, f"Connected to Anthropic ({model_name})."
        elif provider == "Google Gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            genai.GenerativeModel(model_name).generate_content(
                "ping", generation_config={"max_output_tokens": 1}
            )
            return True, f"Connected to Google Gemini ({model_name})."
        else:
            return False, f"Unknown provider: {provider}"
    except Exception as exc:
        return False, str(exc)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Topic Modeling Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Topic Modeling Suite")
st.markdown("Comprehensive topic modeling with LDA, BERTopic, Turftopic, BunkaTopics, and STM.")

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
for key in ["df", "texts", "embeddings", "results", "config", "time_column",
            "chunk_map", "original_texts"]:
    if key not in st.session_state:
        st.session_state[key] = None
if not st.session_state.results:
    st.session_state.results = {}

# ---------------------------------------------------------------------------
# Sidebar - LLM Configuration (always at top)
# ---------------------------------------------------------------------------
with st.sidebar:
    with st.expander("🤖 LLM Topic Labeling", expanded=False):
        use_llm_labeling = st.checkbox(
            "Use LLM to name/describe topics", value=False,
            help="Use an LLM to generate human-readable topic names and descriptions",
        )
        llm_api_key = None
        llm_model_name = "gpt-4o-mini"
        if use_llm_labeling:
            llm_provider = st.selectbox(
                "LLM provider:", ["OpenAI", "Anthropic (Claude)", "Google Gemini"],
            )
            llm_api_key = st.text_input("API key:", type="password")
            if llm_provider == "OpenAI":
                llm_model_name = st.selectbox(
                    "OpenAI model:", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-nano", "gpt-4.1-mini"],
                )
            elif llm_provider == "Anthropic (Claude)":
                llm_model_name = st.selectbox(
                    "Claude model:",
                    ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-7"],
                )
            elif llm_provider == "Google Gemini":
                llm_model_name = st.selectbox(
                    "Gemini model:", ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
                )
            if st.button("🔌 Verify Connection", use_container_width=True):
                with st.spinner("Testing connection..."):
                    ok, msg = _verify_llm_connection(llm_provider, llm_api_key, llm_model_name)
                if ok:
                    st.success(msg)
                else:
                    st.error(f"Connection failed: {msg}")
    st.divider()

# ---------------------------------------------------------------------------
# Sidebar - Data Loading
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Data Source")
    data_source = st.radio("Choose data source:", ["Upload CSV", "Folder Path", "Sample Dataset"])
    df = None

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file:
            df = load_csv(uploaded_file)
    elif data_source == "Folder Path":
        folder_path = st.text_input("Enter folder path containing text files:")
        if folder_path:
            df = load_folder(folder_path)
    elif data_source == "Sample Dataset":
        datasets = get_sample_datasets()
        dataset_name = st.selectbox("Choose sample dataset:", list(datasets.keys()))
        st.caption(datasets[dataset_name])
        if st.button("Load Dataset"):
            with st.spinner("Loading dataset..."):
                df = load_sample_dataset(dataset_name)
                st.success(f"Loaded {len(df)} documents")

    if df is not None and not df.empty:
        st.session_state.df = df
        st.success(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")

# ---------------------------------------------------------------------------
# Column selection and preprocessing
# ---------------------------------------------------------------------------
if st.session_state.df is not None:
    df = st.session_state.df
    with st.sidebar:
        st.header("2. Column Selection")
        text_cols = get_text_columns(df)
        all_cols = list(df.columns)
        selected_text_cols = st.multiselect(
            "Select text column(s) to use as corpus:",
            options=all_cols,
            default=text_cols[:1] if text_cols else all_cols[:1],
        )
        if selected_text_cols:
            separator = st.text_input("Column separator:", value=" ")
            texts_clean, valid_mask = _cached_get_texts(df, tuple(selected_text_cols), separator)
            st.info(f"{len(texts_clean)} valid documents after filtering")

            st.subheader("Preprocessing")
            enable_preprocessing = st.checkbox("Enable text preprocessing", value=False)
            preproc_opts = {}
            if enable_preprocessing:
                preproc_opts["lowercase"] = st.checkbox("Lowercase", value=True)
                preproc_opts["remove_punctuation"] = st.checkbox("Remove punctuation", value=True)
                preproc_opts["remove_numbers"] = st.checkbox("Remove numbers", value=True)
                preproc_opts["remove_stopwords"] = st.checkbox("Remove stopwords", value=True)
                preproc_opts["lemmatize"] = st.checkbox(
                    "Lemmatize (requires spacy)", value=True,
                    help="Reduce words to base form: 'running' → 'run'"
                )
                preproc_opts["content_words_only"] = st.checkbox(
                    "Content words only", value=False,
                    help="Keep only nouns, verbs, adjectives, adverbs"
                )
                custom_sw = st.text_input("Additional stopwords (comma-separated):", value="")
                custom_sw_list = [w.strip() for w in custom_sw.split(",") if w.strip()] if custom_sw.strip() else []
                processed_list = _cached_advanced_preprocess(
                    tuple(texts_clean.tolist()),
                    lowercase=preproc_opts.get("lowercase", False),
                    remove_punctuation=preproc_opts.get("remove_punctuation", False),
                    remove_numbers=preproc_opts.get("remove_numbers", False),
                    remove_stopwords=preproc_opts.get("remove_stopwords", False),
                    lemmatize=preproc_opts.get("lemmatize", False),
                    content_words_only=preproc_opts.get("content_words_only", False),
                    custom_stopwords_tuple=tuple(custom_sw_list),
                )
                texts_clean = pd.Series(processed_list, index=texts_clean.index)
                texts_clean = texts_clean[texts_clean.str.len() >= 5]
                st.info(f"{len(texts_clean)} documents after preprocessing")

            st.subheader("Document Chunking")
            enable_chunking = st.checkbox("Chunk long documents", value=False,
                                          help="Split documents into overlapping chunks")
            if enable_chunking:
                max_words = st.number_input("Max words per chunk:", 100, 5000, 500)
                overlap = st.number_input("Overlap words:", 0, 500, 50)
                original_texts = texts_clean.tolist()
                chunks, doc_indices = _cached_chunk_texts(tuple(original_texts), max_words, overlap)
                st.info(f"{len(original_texts)} docs → {len(chunks)} chunks")
                st.session_state.texts = chunks
                st.session_state.chunk_map = doc_indices
                st.session_state.original_texts = original_texts
            else:
                st.session_state.texts = texts_clean.tolist()
                st.session_state.chunk_map = None
                st.session_state.original_texts = texts_clean.tolist()

            time_col = detect_time_column(df)
            all_potential_time_cols = [c for c in all_cols if c not in selected_text_cols]
            time_col_selected = st.selectbox(
                "Time column (optional, for evolution):",
                options=["None"] + all_potential_time_cols,
                index=0 if time_col is None else all_potential_time_cols.index(time_col) + 1
                if time_col in all_potential_time_cols else 0,
            )
            if time_col_selected != "None":
                try:
                    st.session_state.time_column = pd.to_datetime(
                        df.loc[valid_mask, time_col_selected], errors="coerce"
                    )
                except Exception:
                    st.session_state.time_column = None
            else:
                st.session_state.time_column = None

    with st.expander("📋 Data Preview", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)
        st.write(f"Shape: {df.shape}")
        st.write(df.dtypes)

# ---------------------------------------------------------------------------
# Sidebar - Model Configuration
# ---------------------------------------------------------------------------
if st.session_state.texts:
    with st.sidebar:
        st.header("3. Embedding Model")
        embedding_model = st.selectbox("Choose embedding model:", get_available_models())
        api_key = None
        if requires_api_key(embedding_model):
            api_key = st.text_input(get_api_key_label(embedding_model), type="password")

        st.header("4. Topic Models")
        models_to_run = st.multiselect(
            "Select topic modeling approaches:",
            ["LDA", "BERTopic", "Turftopic", "BunkaTopics", "STM"],
            default=["LDA", "BERTopic"],
        )

        st.header("5. Settings")
        _n_docs = len(st.session_state.texts)
        _max_topics_slider = 100 if _n_docs >= 5000 else 50
        n_topics = st.slider("Number of topics:", min_value=2, max_value=_max_topics_slider, value=8)

        lda_settings = {}
        bertopic_settings = {}
        turftopic_settings = {}
        bunka_settings = {}
        stm_settings = {}

        if "LDA" in models_to_run:
            with st.expander("LDA Settings"):
                st.caption("Uses Gensim LdaMulticore (parallel variational Bayes).")
                if "r_ldatuning_available" not in st.session_state:
                    try:
                        from utils.lda_modeling import _check_ldatuning_available
                        st.session_state.r_ldatuning_available = _check_ldatuning_available()
                    except Exception:
                        st.session_state.r_ldatuning_available = False

                r_available = st.session_state.r_ldatuning_available
                lda_settings["use_ldatuning"] = st.checkbox(
                    "Use R ldatuning (4 metrics, recommended)" if r_available
                    else "Use R ldatuning (R + ldatuning not detected)", value=r_available,
                )
                lda_settings["auto_topics"] = st.checkbox(
                    "Auto-detect optimal topics (Python perplexity/coherence)", value=not r_available,
                )
                n_docs = len(st.session_state.texts)
                if n_docs >= 10000:
                    max_k_upper, max_k_default = 100, 100
                elif n_docs >= 5000:
                    max_k_upper, max_k_default = 100, 50
                elif n_docs >= 1000:
                    max_k_upper, max_k_default = 75, 30
                else:
                    max_k_upper, max_k_default = 50, 20

                lda_settings["min_k"] = st.number_input("Min topics (tuning):", 2, max_k_upper, 2)
                lda_settings["max_k"] = st.number_input(
                    "Max topics (tuning):", 2, max_k_upper, max_k_default,
                    help=f"Corpus: {n_docs} docs. Evaluating up to {max_k_default} topics.",
                )
                lda_settings["max_features"] = st.number_input("Max vocabulary size:", 1000, 50000, 5000)
                lda_settings["passes"] = st.number_input(
                    "Passes (training epochs):", 1, 200, 20,
                    help="Number of full passes through the corpus. More = better quality but slower.",
                )
                lda_settings["iterations"] = st.number_input(
                    "E-step iterations:", 50, 2500, 400,
                    help="Max iterations for document-topic inference per pass.",
                )
                lda_settings["n_top_words"] = st.number_input("Top words per topic:", 5, 30, 15)

        if "BERTopic" in models_to_run:
            with st.expander("BERTopic Settings"):
                bertopic_settings["auto_topics"] = st.checkbox(
                    "Auto-detect number of topics", value=True,
                    help="Let HDBSCAN find the natural clusters",
                )
                bertopic_settings["min_topic_size"] = st.number_input("Min topic size:", 2, 100, 10)
                bertopic_settings["n_gram_range"] = st.selectbox(
                    "N-gram range:", [(1, 1), (1, 2), (1, 3)], index=1,
                    format_func=lambda x: f"{x[0]}-{x[1]}"
                )
                bertopic_settings["reduce_topics"] = (
                    st.checkbox("Reduce to N topics", value=True)
                    if not bertopic_settings["auto_topics"] else False
                )

        if "Turftopic" in models_to_run:
            with st.expander("Turftopic Settings"):
                from utils.turftopic_modeling import TURFTOPIC_MODELS, get_model_descriptions, AUTO_TOPIC_MODELS
                descriptions = get_model_descriptions()
                turftopic_settings["model_type"] = st.selectbox(
                    "Turftopic model:", list(TURFTOPIC_MODELS.keys()),
                )
                st.caption(descriptions.get(turftopic_settings["model_type"], ""))
                model_key = TURFTOPIC_MODELS.get(turftopic_settings["model_type"], "KeyNMF")
                turftopic_settings["auto_topics"] = (
                    st.checkbox("Auto-detect number of topics", value=True)
                    if model_key in AUTO_TOPIC_MODELS else False
                )

        if "BunkaTopics" in models_to_run:
            with st.expander("BunkaTopics Settings"):
                bunka_settings["min_count_terms"] = st.number_input("Min term count:", 1, 20, 2)

        if "STM" in models_to_run:
            with st.expander("STM Settings"):
                metadata_cols = get_metadata_columns(df, selected_text_cols)
                stm_settings["metadata_cols"] = st.multiselect(
                    "Metadata columns for prevalence:", options=metadata_cols,
                )
                if stm_settings["metadata_cols"]:
                    formula_parts = [
                        col if df[col].dtype in ["int64", "float64"] else f"as.factor({col})"
                        for col in stm_settings["metadata_cols"]
                    ]
                    stm_settings["prevalence_formula"] = " + ".join(formula_parts)
                    st.code(f"Prevalence: ~ {stm_settings['prevalence_formula']}")
                else:
                    stm_settings["prevalence_formula"] = None
                stm_settings["max_iter"] = st.number_input("STM max iterations:", 10, 200, 75)

        run_button = st.button("🚀 Run Topic Modeling", type="primary", use_container_width=True)

# ===========================================================================
# ANALYSIS — runs only when the Run button is clicked
# Results are stored in session state so the display block below always works.
# ===========================================================================
if st.session_state.texts and "run_button" in dir() and run_button:
    texts = st.session_state.texts
    all_results = {}

    config = {
        "data_source": data_source,
        "text_columns": selected_text_cols if "selected_text_cols" in dir() else [],
        "embedding_model": embedding_model,
        "n_topics": n_topics,
        "models": models_to_run,
    }
    st.session_state.config = config

    # --- Embeddings ---
    needs_embeddings = any(m in models_to_run for m in ["BERTopic", "Turftopic", "BunkaTopics"])
    embeddings = None
    if needs_embeddings:
        if requires_api_key(embedding_model) and not api_key:
            st.error(f"Please provide {get_api_key_label(embedding_model)}")
            st.stop()
        with st.spinner(f"Computing embeddings with {embedding_model}..."):
            try:
                embeddings = compute_embeddings(texts, embedding_model, api_key)
                st.session_state.embeddings = embeddings
                st.success(f"Embeddings computed: {embeddings.shape}")
            except Exception as e:
                st.error(f"Embedding computation failed: {e}")
                st.stop()

    # --- LDA ---
    if "LDA" in models_to_run:
        from utils.lda_modeling import (
            find_optimal_topics, find_optimal_topics_r, find_optimal_k_ldatuning,
            run_lda_multicore,
        )
        actual_n_topics = n_topics

        ldatuning_results = None
        if lda_settings.get("use_ldatuning", False):
            ldatuning_results = find_optimal_topics_r(
                texts,
                min_topics=lda_settings.get("min_k", 2),
                max_topics=lda_settings.get("max_k", 20),
                step=2,
                max_features=lda_settings.get("max_features", 5000),
            )
            if ldatuning_results is not None:
                actual_n_topics = find_optimal_k_ldatuning(ldatuning_results)
            else:
                st.warning("R ldatuning not available — falling back to manual topic count.")

        tuning = None
        if lda_settings.get("auto_topics", False):
            with st.spinner("Finding optimal number of topics..."):
                tuning = find_optimal_topics(
                    texts,
                    min_topics=lda_settings.get("min_k", 2),
                    max_topics=lda_settings.get("max_k", 20),
                    max_features=lda_settings.get("max_features", 5000),
                )
                if not lda_settings.get("use_ldatuning", False):
                    actual_n_topics = tuning["optimal_topics"]

        try:
            lda_results = run_lda_multicore(
                texts, actual_n_topics,
                max_features=lda_settings.get("max_features", 5000),
                n_top_words=lda_settings.get("n_top_words", 15),
                iterations=lda_settings.get("iterations", 400),
                passes=lda_settings.get("passes", 20),
            )
            st.success("LdaMulticore complete.")

            if st.session_state.chunk_map is not None:
                original_doc_topic = aggregate_chunk_results(
                    lda_results["doc_topic_matrix"],
                    st.session_state.chunk_map,
                    len(st.session_state.original_texts),
                )
                orig_df = pd.DataFrame(
                    original_doc_topic,
                    columns=[f"Topic {i+1}" for i in range(actual_n_topics)]
                )
                orig_df["dominant_topic"] = orig_df.idxmax(axis=1)
                orig_df["dominant_topic_prob"] = orig_df.iloc[:, :actual_n_topics].max(axis=1)
                orig_df["text_preview"] = [
                    t[:100] + "..." if len(t) > 100 else t for t in st.session_state.original_texts
                ]
                lda_results["doc_topic_df_aggregated"] = orig_df
                lda_results["doc_topic_matrix_aggregated"] = original_doc_topic

            lda_results["ldatuning_results"] = ldatuning_results
            lda_results["tuning"] = tuning
            all_results["lda"] = lda_results
        except Exception as lda_err:
            st.error(f"LDA (LdaMulticore) failed: {lda_err}")

    # --- BERTopic ---
    if "BERTopic" in models_to_run:
        from utils.bertopic_modeling import (
            run_bertopic, label_topics_with_llm as bertopic_llm_label,
        )
        from utils.embeddings import get_embedding_model

        emb_model = None
        if "SBERT" in embedding_model:
            emb_model = get_embedding_model(embedding_model)

        bt_results = run_bertopic(
            texts, embeddings=embeddings, embedding_model=emb_model,
            n_topics=n_topics,
            min_topic_size=bertopic_settings.get("min_topic_size", 10),
            n_gram_range=bertopic_settings.get("n_gram_range", (1, 2)),
            nr_topics=n_topics if bertopic_settings.get("reduce_topics", False) else None,
            auto_topics=bertopic_settings.get("auto_topics", True),
        )
        if use_llm_labeling and llm_api_key:
            with st.spinner("BERTopic: Generating LLM topic labels..."):
                bt_results = bertopic_llm_label(bt_results, llm_api_key, llm_model_name)
        all_results["bertopic"] = bt_results

    # --- Turftopic ---
    if "Turftopic" in models_to_run:
        from utils.turftopic_modeling import (
            TURFTOPIC_MODELS, run_turftopic, label_topics_with_llm as tt_llm_label,
        )
        model_display = turftopic_settings.get("model_type", "KeyNMF")
        model_key = TURFTOPIC_MODELS.get(model_display, "KeyNMF")
        encoder = "all-mpnet-base-v2" if ("SBERT" in embedding_model and "mpnet" in embedding_model) else "all-MiniLM-L6-v2"
        try:
            tt_results = run_turftopic(
                texts, model_type=model_key, n_topics=n_topics, encoder_model=encoder,
                auto_topics=turftopic_settings.get("auto_topics", False),
            )
            if use_llm_labeling and llm_api_key:
                with st.spinner("Turftopic: Generating LLM topic labels..."):
                    tt_results = tt_llm_label(tt_results, llm_api_key, llm_model_name)
            all_results["turftopic"] = tt_results
        except Exception as e:
            st.error(f"Turftopic failed: {e}")

    # --- BunkaTopics ---
    if "BunkaTopics" in models_to_run:
        from utils.bunka_modeling import (
            run_bunkatopics, label_topics_with_llm as bk_llm_label,
        )
        bk_emb_model = "all-mpnet-base-v2" if ("SBERT" in embedding_model and "mpnet" in embedding_model) else "all-MiniLM-L6-v2"
        try:
            bk_results = run_bunkatopics(
                texts, n_topics=n_topics, embedding_model=bk_emb_model,
                min_count_terms=bunka_settings.get("min_count_terms", 2),
            )
            if use_llm_labeling and llm_api_key:
                with st.spinner("BunkaTopics: Generating LLM topic labels..."):
                    bk_results = bk_llm_label(bk_results, llm_api_key, llm_model_name)
            all_results["bunkatopics"] = bk_results
        except Exception as e:
            st.error(f"BunkaTopics failed: {e}")

    # --- STM ---
    if "STM" in models_to_run:
        from utils.stm_modeling import (
            check_r_available, run_stm, run_stm_python_fallback,
        )
        stm_metadata_df = None
        if stm_settings.get("metadata_cols"):
            stm_metadata_df = df[stm_settings["metadata_cols"]].iloc[:len(texts)].copy()
        try:
            r_avail = check_r_available()
            if r_avail:
                stm_results = run_stm(
                    texts, metadata_df=stm_metadata_df,
                    prevalence_formula=stm_settings.get("prevalence_formula"),
                    n_topics=n_topics,
                    max_iter=stm_settings.get("max_iter", 75),
                )
            else:
                st.warning("R not available — using Python NMF approximation for STM.")
                stm_results = run_stm_python_fallback(texts, n_topics=n_topics)
            stm_results["r_used"] = r_avail
            all_results["stm"] = stm_results
        except Exception as e:
            st.error(f"STM failed: {e}")

    st.session_state.results = all_results
    if all_results:
        st.success("✅ Analysis complete — results are shown below.")

# ===========================================================================
# DISPLAY — always rendered from session state, survives any button click
# ===========================================================================
if st.session_state.results:
    all_results = st.session_state.results
    _cfg = st.session_state.config or {}
    _emb_model = _cfg.get("embedding_model", "SBERT (all-MiniLM-L6-v2)")

    # -----------------------------------------------------------------------
    # LDA
    # -----------------------------------------------------------------------
    if "lda" in all_results:
        from utils.lda_modeling import (
            get_optimization_explanation, plot_ldatuning_results, find_optimal_k_ldatuning,
            plot_topic_tuning, plot_topic_words, plot_topic_distribution,
            plot_topic_correlations, plot_doc_topic_heatmap,
            plot_topic_evolution, generate_pyldavis, plot_word_clouds,
        )
        lda_results = all_results["lda"]

        st.header("📈 LDA Topic Modeling")
        st.caption("Gensim LdaMulticore (parallel variational Bayes)")

        with st.expander("ℹ️ How is the optimal number of topics determined?"):
            st.markdown(get_optimization_explanation())

        # Show stored optimization artifacts
        if lda_results.get("ldatuning_results") is not None:
            ld = lda_results["ldatuning_results"]
            optimal_k = find_optimal_k_ldatuning(ld)
            st.plotly_chart(plot_ldatuning_results(ld, optimal_k=optimal_k), use_container_width=True)
            st.dataframe(ld["results"], use_container_width=True)
            st.success(f"Optimal topics (R ldatuning composite): **{optimal_k}**")

        if lda_results.get("tuning") is not None:
            tuning = lda_results["tuning"]
            st.plotly_chart(plot_topic_tuning(tuning), use_container_width=True)
            st.info(f"Python elbow method suggests: **{tuning['optimal_topics']}** topics")

        _perp = lda_results["perplexity"]
        st.metric("Perplexity", f"{_perp:.2f}" if np.isfinite(_perp) else "N/A")

        tab_words, tab_dist, tab_corr, tab_heatmap, tab_wc, tab_pyldavis, tab_evo = st.tabs([
            "Top Words", "Distribution", "Correlations", "Doc-Topic Matrix",
            "Word Clouds", "pyLDAvis", "Evolution",
        ])
        with tab_words:
            st.plotly_chart(plot_topic_words(lda_results), use_container_width=True)
        with tab_dist:
            st.plotly_chart(plot_topic_distribution(lda_results), use_container_width=True)
        with tab_corr:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_topic_correlations(lda_results), use_container_width=True)
            with col2:
                from utils.visualization import plot_topic_similarity_network
                st.plotly_chart(
                    plot_topic_similarity_network(lda_results["topic_correlations"], title="LDA Topic Network"),
                    use_container_width=True,
                )
        with tab_heatmap:
            max_docs = st.slider("Documents to show:", 10, 200, 50, key="lda_heatmap_docs")
            st.plotly_chart(plot_doc_topic_heatmap(lda_results, max_docs), use_container_width=True)
        with tab_wc:
            wc_figs = plot_word_clouds(lda_results)
            wc_cols = st.columns(min(3, len(wc_figs)))
            for i, fig in enumerate(wc_figs):
                with wc_cols[i % len(wc_cols)]:
                    st.pyplot(fig)
        with tab_pyldavis:
            html = generate_pyldavis(lda_results)
            if html:
                import streamlit.components.v1 as components
                components.html(html, height=800, scrolling=True)
            else:
                st.info("pyLDAvis not available. Install pyLDAvis: pip install pyldavis")
        with tab_evo:
            if st.session_state.time_column is not None:
                freq = st.selectbox("Time frequency:", ["Y", "Q", "M"], key="lda_freq")
                fig = plot_topic_evolution(lda_results, st.session_state.time_column, freq)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No time column detected.")

        with st.expander("📋 Document-Topic Assignments"):
            display_df = lda_results.get("doc_topic_df_aggregated", lda_results["doc_topic_df"])
            if st.session_state.chunk_map is not None and "doc_topic_df_aggregated" in lda_results:
                st.caption("Aggregated from chunk-level results.")
            st.dataframe(display_df, use_container_width=True)

    # -----------------------------------------------------------------------
    # BERTopic
    # -----------------------------------------------------------------------
    if "bertopic" in all_results:
        from utils.bertopic_modeling import (
            plot_topics_barchart, plot_topics_scatter, plot_hierarchy, plot_heatmap,
            plot_documents, plot_topic_evolution as bertopic_evolution,
            plot_topic_correlations as bertopic_corr,
            get_representative_docs, get_topic_tree,
        )
        bt_results = all_results["bertopic"]
        st.header("🧠 BERTopic")

        n_outliers = bt_results["topic_info"].iloc[0]["Count"] if -1 in bt_results["topics"] else 0
        st.write(f"Found **{len(bt_results['topic_info']) - 1}** topics ({n_outliers} outlier docs)")
        st.dataframe(bt_results["topic_info"].head(20), use_container_width=True)

        if "llm_topic_names" in bt_results:
            with st.expander("🤖 LLM-Generated Topic Labels", expanded=True):
                st.dataframe(pd.DataFrame({
                    "Topic": [f"Topic {i}" for i in range(len(bt_results["llm_topic_names"]))],
                    "LLM Name": bt_results["llm_topic_names"],
                    "LLM Description": bt_results.get("llm_topic_descriptions", []),
                }), use_container_width=True)

        tab_bar, tab_scatter, tab_hier, tab_heat, tab_docs, tab_tree, tab_corr_bt, tab_evo_bt = st.tabs([
            "Barchart", "Intertopic Map", "Hierarchy", "Heatmap",
            "Documents", "Topic Tree", "Correlations", "Evolution",
        ])
        with tab_bar:
            fig = plot_topics_barchart(bt_results)
            if fig: st.plotly_chart(fig, use_container_width=True)
        with tab_scatter:
            fig = plot_topics_scatter(bt_results)
            if fig: st.plotly_chart(fig, use_container_width=True)
        with tab_hier:
            fig = plot_hierarchy(bt_results)
            if fig: st.plotly_chart(fig, use_container_width=True)
        with tab_heat:
            fig = plot_heatmap(bt_results)
            if fig: st.plotly_chart(fig, use_container_width=True)
        with tab_docs:
            fig = plot_documents(bt_results)
            if fig: st.plotly_chart(fig, use_container_width=True)
        with tab_tree:
            tree = get_topic_tree(bt_results)
            st.text(tree) if tree else st.info("Topic tree not available.")
        with tab_corr_bt:
            fig = bertopic_corr(bt_results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Correlations require calculate_probabilities=True.")
        with tab_evo_bt:
            if st.session_state.time_column is not None:
                nr_bins = st.slider("Number of time bins:", 5, 30, 10, key="bt_bins")
                fig, topics_over_time = bertopic_evolution(bt_results, st.session_state.time_column, nr_bins)
                if fig: st.plotly_chart(fig, use_container_width=True)
                if topics_over_time is not None:
                    with st.expander("Topics over time data"):
                        st.dataframe(topics_over_time, use_container_width=True)
            else:
                st.info("No time column detected.")

        with st.expander("📄 Representative Documents per Topic"):
            topic_ids = sorted([t for t in set(bt_results["topics"]) if t != -1])
            for tid in topic_ids[:10]:
                docs = get_representative_docs(bt_results, tid, n=3)
                if docs:
                    st.write(f"**Topic {tid}:**")
                    for d in docs:
                        st.write(f"- {d[:200]}...")
                    st.divider()
        with st.expander("📋 Document-Topic Assignments"):
            st.dataframe(bt_results["doc_topic_df"], use_container_width=True)

    # -----------------------------------------------------------------------
    # Turftopic
    # -----------------------------------------------------------------------
    if "turftopic" in all_results:
        from utils.turftopic_modeling import (
            plot_topic_words as tt_words, plot_topic_distribution as tt_dist,
            plot_topic_correlations as tt_corr, plot_doc_topic_heatmap as tt_heatmap,
            plot_topic_evolution as tt_evo,
        )
        tt_results = all_results["turftopic"]
        st.header("🌊 Turftopic")

        tt_tab_names = ["Top Words", "Distribution", "Correlations", "Doc-Topic Matrix", "Evolution"]
        if "llm_topic_names" in tt_results:
            tt_tab_names.append("LLM Labels")
        tt_tabs = st.tabs(tt_tab_names)

        with tt_tabs[0]: st.plotly_chart(tt_words(tt_results), use_container_width=True)
        with tt_tabs[1]: st.plotly_chart(tt_dist(tt_results), use_container_width=True)
        with tt_tabs[2]:
            fig = tt_corr(tt_results)
            if fig: st.plotly_chart(fig, use_container_width=True)
        with tt_tabs[3]:
            max_docs = st.slider("Documents to show:", 10, 200, 50, key="tt_heatmap_docs")
            st.plotly_chart(tt_heatmap(tt_results, max_docs), use_container_width=True)
        with tt_tabs[4]:
            if st.session_state.time_column is not None:
                freq = st.selectbox("Time frequency:", ["Y", "Q", "M"], key="tt_freq")
                fig = tt_evo(tt_results, st.session_state.time_column, freq)
                if fig: st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No time column detected.")
        if "llm_topic_names" in tt_results:
            with tt_tabs[5]:
                st.dataframe(pd.DataFrame({
                    "Topic": [f"Topic {i+1}" for i in range(len(tt_results["llm_topic_names"]))],
                    "LLM Name": tt_results["llm_topic_names"],
                    "LLM Description": tt_results.get("llm_topic_descriptions", [""] * len(tt_results["llm_topic_names"])),
                }), use_container_width=True)

        with st.expander("📋 Document-Topic Assignments"):
            st.dataframe(tt_results["doc_topic_df"], use_container_width=True)

    # -----------------------------------------------------------------------
    # BunkaTopics
    # -----------------------------------------------------------------------
    if "bunkatopics" in all_results:
        from utils.bunka_modeling import (
            plot_bunka_map, plot_bunka_docs,
            plot_topic_words as bk_words, plot_topic_distribution as bk_dist,
        )
        bk_results = all_results["bunkatopics"]
        st.header("🗺️ BunkaTopics")

        bk_tab_names = ["Topic Map", "Document Map", "Top Words", "Distribution"]
        if "llm_topic_names" in bk_results:
            bk_tab_names.append("LLM Labels")
        bk_tabs = st.tabs(bk_tab_names)

        with bk_tabs[0]:
            fig = plot_bunka_map(bk_results)
            if fig: st.plotly_chart(fig, use_container_width=True)
        with bk_tabs[1]:
            fig = plot_bunka_docs(bk_results)
            if fig: st.plotly_chart(fig, use_container_width=True)
        with bk_tabs[2]: st.plotly_chart(bk_words(bk_results), use_container_width=True)
        with bk_tabs[3]: st.plotly_chart(bk_dist(bk_results), use_container_width=True)
        if "llm_topic_names" in bk_results:
            with bk_tabs[4]:
                st.dataframe(pd.DataFrame({
                    "Topic": list(bk_results["topics"].keys())[:len(bk_results["llm_topic_names"])],
                    "LLM Name": bk_results["llm_topic_names"],
                    "LLM Description": bk_results.get("llm_topic_descriptions", [""] * len(bk_results["llm_topic_names"])),
                }), use_container_width=True)

        with st.expander("📋 Document-Topic Assignments"):
            st.dataframe(bk_results["doc_topic_df"], use_container_width=True)

    # -----------------------------------------------------------------------
    # STM
    # -----------------------------------------------------------------------
    if "stm" in all_results:
        from utils.stm_modeling import (
            plot_topic_words as stm_words, plot_topic_correlations as stm_corr,
            plot_quality, plot_topic_distribution as stm_dist,
            plot_topic_evolution as stm_evo,
        )
        stm_results = all_results["stm"]
        r_used = stm_results.get("r_used", False)
        st.header("📐 Structural Topic Modeling")

        tabs_list = ["Top Words", "Distribution", "Correlations", "Quality", "Evolution"]
        if r_used and stm_results.get("r_script"):
            tabs_list.append("R Script")
        if stm_results.get("prevalence_effects_summary"):
            tabs_list.append("Prevalence Effects")

        stm_tabs = st.tabs(tabs_list)
        with stm_tabs[0]: st.plotly_chart(stm_words(stm_results), use_container_width=True)
        with stm_tabs[1]: st.plotly_chart(stm_dist(stm_results), use_container_width=True)
        with stm_tabs[2]:
            st.plotly_chart(stm_corr(stm_results), use_container_width=True)
            from utils.visualization import plot_topic_similarity_network
            st.plotly_chart(
                plot_topic_similarity_network(stm_results["topic_correlations"], title="STM Topic Network"),
                use_container_width=True,
            )
        with stm_tabs[3]:
            if any(c != 0 for c in stm_results.get("coherence", [])):
                st.plotly_chart(plot_quality(stm_results), use_container_width=True)
            else:
                st.info("Quality metrics not available (Python fallback mode).")
        with stm_tabs[4]:
            if st.session_state.time_column is not None:
                freq = st.selectbox("Time frequency:", ["Y", "Q", "M"], key="stm_freq")
                fig = stm_evo(stm_results, st.session_state.time_column, freq)
                if fig: st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No time column detected.")
        if r_used and stm_results.get("r_script") and "R Script" in tabs_list:
            with stm_tabs[tabs_list.index("R Script")]:
                st.markdown("### Standalone R Script")
                st.code(stm_results["r_script"], language="r")
                st.download_button("⬇️ Download R Script", stm_results["r_script"],
                                   "stm_analysis.R", "text/plain", key="dl_r_script")
        if stm_results.get("prevalence_effects_summary") and "Prevalence Effects" in tabs_list:
            with stm_tabs[tabs_list.index("Prevalence Effects")]:
                st.markdown("### Prevalence Effects Summary")
                st.text(stm_results["prevalence_effects_summary"])

        with st.expander("📋 Document-Topic Assignments"):
            st.dataframe(stm_results["doc_topic_df"], use_container_width=True)

    # -----------------------------------------------------------------------
    # Cross-model comparison
    # -----------------------------------------------------------------------
    if len(all_results) > 1:
        st.header("📊 Cross-Model Comparison")
        from utils.visualization import plot_topic_comparison, create_summary_table
        st.plotly_chart(plot_topic_comparison(all_results), use_container_width=True)
        summary_dfs = [create_summary_table(r, m) for m, r in all_results.items()]
        if summary_dfs:
            st.dataframe(pd.concat(summary_dfs, ignore_index=True), use_container_width=True)

    # -----------------------------------------------------------------------
    # Master Topic List
    # -----------------------------------------------------------------------
    st.header("📋 Master Topic List")
    st.markdown("Non-overlapping topics merged across all models using semantic similarity.")
    from utils.topic_masterlist import build_masterlist, plot_topic_similarity_heatmap, plot_cluster_network

    sim_threshold = st.slider(
        "Similarity threshold for merging:", min_value=0.3, max_value=0.95,
        value=0.65, step=0.05, key="masterlist_threshold",
    )
    masterlist_emb = "all-mpnet-base-v2" if ("SBERT" in _emb_model and "mpnet" in _emb_model) else "all-MiniLM-L6-v2"
    masterlist_result = build_masterlist(all_results, similarity_threshold=sim_threshold, embedding_model=masterlist_emb)
    masterlist_df = masterlist_result["masterlist_df"]

    if not masterlist_df.empty:
        st.success(f"**{len(masterlist_df)}** master topics from {len(all_results)} model(s)")
        display_cols = ["Master Topic", "Top Words", "Models", "Model Count"]
        if masterlist_df["LLM Label"].any():
            display_cols[2:2] = ["LLM Label", "LLM Description"]
        st.dataframe(masterlist_df[display_cols], use_container_width=True, hide_index=True)

        with st.expander("Source Topics per Master Topic"):
            st.dataframe(masterlist_df[["Master Topic", "Source Topics", "Top Words"]],
                         use_container_width=True, hide_index=True)

        tab_sim, tab_net = st.tabs(["Similarity Heatmap", "Cluster Network"])
        with tab_sim:
            fig = plot_topic_similarity_heatmap(masterlist_result)
            if fig: st.plotly_chart(fig, use_container_width=True)
        with tab_net:
            fig = plot_cluster_network(masterlist_result, threshold=sim_threshold)
            if fig: st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "⬇️ Download Master Topic List (CSV)",
            masterlist_df.to_csv(index=False), "master_topic_list.csv", "text/csv",
            key="dl_masterlist",
        )
    st.session_state["masterlist_result"] = masterlist_result

# ===========================================================================
# DOWNLOAD — always available once results exist
# ===========================================================================
if st.session_state.results:
    st.header("📥 Download Results")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Generate Download Package"):
            with st.spinner("Generating download package (including HTML visualizations)..."):
                config = st.session_state.config or {}
                notebook_str = generate_notebook(config, st.session_state.results)
                zip_bytes = create_download_zip(st.session_state.results, config, notebook_str)
                st.session_state["zip_bytes"] = zip_bytes
                st.session_state["notebook_str"] = notebook_str
                st.success("Download package ready!")

    if st.session_state.get("zip_bytes"):
        with col1:
            st.download_button(
                "⬇️ Download All Results (ZIP)",
                data=st.session_state["zip_bytes"],
                file_name="topic_modeling_results.zip",
                mime="application/zip",
            )
        with col2:
            st.download_button(
                "📓 Download Jupyter Notebook",
                data=st.session_state["notebook_str"],
                file_name="topic_modeling_analysis.ipynb",
                mime="application/json",
            )

    with st.expander("Download Individual Results"):
        for model_name, model_results in st.session_state.results.items():
            st.subheader(model_name.upper())
            if "doc_topic_df" in model_results:
                st.download_button(
                    f"📊 {model_name} - Document-Topic Matrix",
                    model_results["doc_topic_df"].to_csv(index=False),
                    f"{model_name}_doc_topic_matrix.csv", "text/csv",
                    key=f"dl_{model_name}_dtm",
                )
            if "topic_info" in model_results:
                st.download_button(
                    f"📋 {model_name} - Topic Info",
                    model_results["topic_info"].to_csv(index=False),
                    f"{model_name}_topic_info.csv", "text/csv",
                    key=f"dl_{model_name}_info",
                )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption("Topic Modeling Suite | LDA, BERTopic, Turftopic, BunkaTopics, STM")
