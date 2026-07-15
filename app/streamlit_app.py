"""
Personalized News Recommender — Streamlit demo app (final version).

Design principles for this version:
  - No dataframe column name is hardcoded. Every column (Title, Abstract,
    Category, NewsID, UserID, History) is auto-detected from whatever the
    processed CSVs actually contain, via `_detect_column()`.
  - No src/ function is assumed to exist without a runtime check. Anything
    missing degrades to a warning + fallback, never a crash.
  - All paths/constants come from src/config.py — nothing is hardcoded here.
"""

import sys
import platform
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src import config

# ---------------------------------------------------------------------------
# Import project functions defensively — missing ones degrade gracefully
# instead of crashing the whole app.
# ---------------------------------------------------------------------------
IMPORT_WARNINGS = []


def _safe_import(module_path, name):
    try:
        module = __import__(module_path, fromlist=[name])
        return getattr(module, name)
    except (ImportError, AttributeError) as e:
        IMPORT_WARNINGS.append(f"Could not import `{name}` from `{module_path}`: {e}")
        return None


load_embeddings = _safe_import("src.embedding", "load_embeddings")
load_news_id_index = _safe_import("src.user_profile", "load_news_id_index")
get_user_vector = _safe_import("src.user_profile", "get_user_vector")
load_index = _safe_import("src.faiss_index", "load_index")
recommend_candidates = _safe_import("src.recommender", "recommend_candidates")
rank_candidates_heuristic = _safe_import("src.ranking", "rank_candidates_heuristic")
rank_candidates_ml = _safe_import("src.ranking", "rank_candidates_ml")
most_similar_history_article = _safe_import("src.explainability", "most_similar_history_article")
load_ranker_model = _safe_import("src.ranker", "load_model")


# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Personalized News Recommender",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }

    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #a855f7 100%);
        border-radius: 18px;
        padding: 36px 40px;
        margin-bottom: 24px;
    }
    .hero-title { color: #fff; font-size: 2.1rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
    .hero-sub { color: #e9e5ff; font-size: 1rem; margin-top: 8px; }
    .pill {
        display: inline-block; background: rgba(255,255,255,0.18); color: #fff;
        padding: 5px 14px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;
        margin-right: 8px; margin-top: 12px;
    }

    .history-chip {
        display: inline-block; background: #1b1f2a; border: 1px solid #2e3440;
        color: #d4d8e0; padding: 6px 12px; border-radius: 10px; font-size: 0.8rem;
        margin: 4px 4px 4px 0;
    }
    .history-chip b { color: #a5b4fc; }

    .rec-card {
        background: #151922; border: 1px solid #232836; border-radius: 14px;
        padding: 18px 20px; margin-bottom: 16px; position: relative;
    }
    .rec-card:hover { border-color: #7c3aed; }
    .rec-rank {
        position: absolute; top: -12px; left: 16px; background: #7c3aed; color: white;
        font-size: 0.72rem; font-weight: 700; padding: 3px 10px; border-radius: 10px;
    }
    .rec-category {
        display: inline-block; background: #232836; color: #93c5fd; font-size: 0.7rem;
        font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;
        padding: 3px 10px; border-radius: 12px; margin-top: 6px;
    }
    .rec-title { color: #f3f4f6; font-size: 1.02rem; font-weight: 700; margin: 8px 0 6px 0; line-height: 1.35; }
    .rec-abstract { color: #a8adba; font-size: 0.85rem; line-height: 1.5; margin-bottom: 10px; }

    .score-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
    .score-chip {
        background: #1f2937; border: 1px solid #2f394a; color: #6ee7b7;
        font-size: 0.72rem; font-weight: 600; padding: 3px 9px; border-radius: 8px;
    }
    .score-chip.blue { color: #93c5fd; }
    .score-chip.purple { color: #d8b4fe; }
    .score-chip.amber { color: #fcd34d; }

    .rec-explain {
        color: #fbbf24; font-size: 0.78rem; font-style: italic; margin-top: 8px;
        border-top: 1px solid #232836; padding-top: 8px;
    }

    .metric-box { background: #151922; border: 1px solid #232836; border-radius: 12px; padding: 16px; text-align: center; }
    .metric-num { font-size: 1.7rem; font-weight: 800; color: #a855f7; }
    .metric-lbl { color: #8b90a0; font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px; }

    .pipeline-step {
        background: #151922; border: 1px solid #232836; border-radius: 12px;
        padding: 14px 20px; text-align: center; font-weight: 700; color: #e5e7eb;
        margin-bottom: 4px;
    }
    .pipeline-arrow { text-align: center; color: #7c3aed; font-size: 1.3rem; margin: 0; }

    footer, .footer-box { color: #6b7280; font-size: 0.78rem; }
    section[data-testid="stSidebar"] { background-color: #10131a; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Column auto-detection — never assume a literal column name.
# ---------------------------------------------------------------------------
COLUMN_CANDIDATES = {
    "NewsID": ["NewsID", "newsid", "news_id", "id", "ID", "News_Id"],
    "Title": ["Title", "title", "NewsTitle", "news_title", "headline", "Headline"],
    "Abstract": ["Abstract", "abstract", "summary", "Summary", "description", "Description"],
    "Category": ["Category", "category", "cat", "Cat", "topic", "Topic"],
    "SubCategory": ["SubCategory", "subcategory", "sub_category", "subcat"],
    "UserID": ["UserID", "userid", "user_id", "uid", "UID"],
    "History": ["History", "history", "clicked_history", "click_history", "read_history"],
}


def _detect_column(df: pd.DataFrame, logical_name: str) -> str:
    """
    Find the actual column in `df` matching a logical field name
    (e.g. 'Title'), trying known aliases first, then a case-insensitive
    substring match, and returning None if nothing is found.
    """
    if df is None:
        return None
    candidates = COLUMN_CANDIDATES.get(logical_name, [logical_name])
    for cand in candidates:
        if cand in df.columns:
            return cand
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for c in df.columns:
        if logical_name.lower() in c.lower():
            return c
    return None


def detect_news_columns(news_df: pd.DataFrame) -> dict:
    cols = {name: _detect_column(news_df, name) for name in ["NewsID", "Title", "Abstract", "Category", "SubCategory"]}
    missing = [k for k, v in cols.items() if v is None and k in ("NewsID", "Title")]
    if missing:
        st.warning(f"Could not auto-detect required column(s) in news data: {missing}. Some features may be limited.")
    return cols


def detect_behavior_columns(behaviors_df: pd.DataFrame) -> dict:
    cols = {name: _detect_column(behaviors_df, name) for name in ["UserID", "History"]}
    missing = [k for k, v in cols.items() if v is None]
    if missing:
        st.warning(f"Could not auto-detect column(s) in behaviors data: {missing}. Some features may be limited.")
    return cols


def get(row, col_map, logical_name, default=""):
    """Safely read a logical field off a row/namedtuple using the detected column map."""
    actual = col_map.get(logical_name)
    if actual is None:
        return default
    val = getattr(row, actual, default) if hasattr(row, "_fields") else row.get(actual, default)
    return val if val is not None else default


# ---------------------------------------------------------------------------
# Cached artifact loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_tabular_artifacts():
    news_df = behaviors_df = popularity_df = None
    warnings = []

    try:
        news_df = pd.read_csv(config.PROCESSED_NEWS)
    except Exception as e:
        warnings.append(f"Could not load PROCESSED_NEWS ({config.PROCESSED_NEWS}): {e}")

    try:
        behaviors_df = pd.read_csv(config.PROCESSED_BEHAVIORS)
    except Exception as e:
        warnings.append(f"Could not load PROCESSED_BEHAVIORS ({config.PROCESSED_BEHAVIORS}): {e}")

    try:
        popularity_df = pd.read_csv(config.POPULARITY_SCORES)
    except Exception as e:
        warnings.append(f"Could not load POPULARITY_SCORES ({config.POPULARITY_SCORES}): {e}")

    return news_df, behaviors_df, popularity_df, warnings


@st.cache_resource(show_spinner=False)
def load_heavy_artifacts():
    news_embeddings = news_id_to_idx = index = None
    logreg_model = lightgbm_model = None
    warnings = []

    try:
        news_embeddings = load_embeddings(config.NEWS_EMBEDDINGS) if load_embeddings else None
    except Exception as e:
        warnings.append(f"Could not load news embeddings: {e}")

    try:
        news_id_to_idx = load_news_id_index(config.NEWS_ID_TO_IDX_PATH) if load_news_id_index else None
    except Exception as e:
        warnings.append(f"Could not load NewsID index: {e}")

    try:
        index = load_index(config.FAISS_INDEX_PATH) if load_index else None
    except Exception as e:
        warnings.append(f"Could not load FAISS index: {e}")

    if load_ranker_model:
        try:
            logreg_model = load_ranker_model(config.LOGREG_MODEL_PATH, is_lightgbm=False)
        except FileNotFoundError:
            pass
        except Exception as e:
            warnings.append(f"Error loading Logistic Regression model: {e}")

        try:
            lightgbm_model = load_ranker_model(config.LIGHTGBM_MODEL_PATH, is_lightgbm=True)
        except FileNotFoundError:
            pass
        except Exception as e:
            warnings.append(f"Error loading LightGBM model: {e}")

    return news_embeddings, news_id_to_idx, index, logreg_model, lightgbm_model, warnings


with st.spinner("Loading recommender artifacts…"):
    try:
        news_df, behaviors_df, popularity_df, tab_warnings = load_tabular_artifacts()
        news_embeddings, news_id_to_idx, index, logreg_model, lightgbm_model, heavy_warnings = load_heavy_artifacts()
    except Exception as e:
        st.error(f"Unexpected failure while loading artifacts: {e}")
        st.code(traceback.format_exc())
        st.stop()

if news_df is None:
    st.error(
        "`processed_news.csv` could not be loaded. The app cannot continue without it. "
        f"Expected path: `{config.PROCESSED_NEWS}`."
    )
    st.stop()

NEWS_COLS = detect_news_columns(news_df)
BEHAVIOR_COLS = detect_behavior_columns(behaviors_df) if behaviors_df is not None else {}

ALL_WARNINGS = IMPORT_WARNINGS + tab_warnings + heavy_warnings


# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <p class="hero-title">📰 Personalized News Recommendation System</p>
        <p class="hero-sub">A two-stage recommender built on Microsoft's MIND dataset.</p>
        <span class="pill">Sentence Transformer</span>
        <span class="pill">FAISS Retrieval</span>
        <span class="pill">Machine Learning Ranking</span>
        <span class="pill">Explainable AI</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if ALL_WARNINGS:
    with st.expander(f"⚠️ {len(ALL_WARNINGS)} warning(s) detected during startup (app still running)"):
        for w in ALL_WARNINGS:
            st.write("- " + w)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Controls")

user_col = BEHAVIOR_COLS.get("UserID")
history_col = BEHAVIOR_COLS.get("History")
can_use_real_users = behaviors_df is not None and user_col and history_col

mode_options = ["Simulate new user"]
if can_use_real_users:
    mode_options.insert(0, "Select existing user")

mode = st.sidebar.radio("User", mode_options)

available_strategies = ["Similarity (heuristic)"] if rank_candidates_heuristic else []
if logreg_model is not None and rank_candidates_ml:
    available_strategies.append("Logistic Regression")
if lightgbm_model is not None and rank_candidates_ml:
    available_strategies.append("LightGBM")
if not available_strategies:
    available_strategies = ["Popularity fallback only"]

strategy = st.sidebar.selectbox("Ranking strategy", available_strategies)
top_k = st.sidebar.slider("Top-K recommendations", 3, 15, 8)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Candidate pool size (FAISS retrieval): **{getattr(config, 'CANDIDATE_POOL_SIZE', 'N/A')}** "
    "articles are retrieved before re-ranking."
)

if logreg_model is None:
    st.sidebar.caption("ℹ️ Logistic Regression model file not found — option hidden.")
if lightgbm_model is None:
    st.sidebar.caption("ℹ️ LightGBM model file not found — option hidden.")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_home, tab_recs, tab_arch, tab_about = st.tabs(["🏠 Home", "✨ Recommendations", "🧩 Architecture", "ℹ️ About"])


# ---------------------------------------------------------------------------
# HOME TAB — statistics + search
# ---------------------------------------------------------------------------
with tab_home:
    st.subheader("📊 Dataset Statistics")

    n_articles = len(news_df) if news_df is not None else 0
    n_users = behaviors_df[user_col].nunique() if (behaviors_df is not None and user_col) else "N/A"
    n_categories = news_df[NEWS_COLS["Category"]].nunique() if NEWS_COLS.get("Category") else "N/A"
    n_embeddings = news_embeddings.shape[0] if news_embeddings is not None else "N/A"
    pool_size = getattr(config, "CANDIDATE_POOL_SIZE", "N/A")

    s1, s2, s3, s4, s5 = st.columns(5)
    for col, (val, label) in zip(
        [s1, s2, s3, s4, s5],
        [
            (n_articles, "Articles"),
            (n_users, "Users"),
            (n_categories, "Categories"),
            (n_embeddings, "Embeddings"),
            (pool_size, "Candidate Pool Size"),
        ],
    ):
        with col:
            st.markdown(
                f'<div class="metric-box"><div class="metric-num">{val}</div>'
                f'<div class="metric-lbl">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🔎 Search News")

    title_col = NEWS_COLS.get("Title")
    if title_col is None:
        st.warning("No title-like column detected in the news data — search is unavailable.")
    else:
        query = st.text_input("Search by article title", placeholder="Type a keyword…")
        if query:
            mask = news_df[title_col].astype(str).str.contains(query, case=False, na=False)
            results = news_df[mask].head(20)
            if results.empty:
                st.info("No matching articles found.")
            else:
                options = results[title_col].astype(str).tolist()
                chosen_title = st.selectbox("Matching articles", options)
                chosen_row = results[results[title_col].astype(str) == chosen_title].iloc[0]

                st.markdown("#### Article Details")
                st.write(f"**Title:** {get(chosen_row, NEWS_COLS, 'Title', 'N/A')}")
                st.write(f"**Category:** {get(chosen_row, NEWS_COLS, 'Category', 'N/A')}")
                st.write(f"**NewsID:** {get(chosen_row, NEWS_COLS, 'NewsID', 'N/A')}")
                abstract_val = get(chosen_row, NEWS_COLS, "Abstract", "")
                st.write(f"**Abstract:** {abstract_val if str(abstract_val).strip() else 'No abstract available.'}")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🖥️ System Information"):
        st.write(f"**Python version:** {platform.python_version()}")
        st.write(f"**Embedding model:** {getattr(config, 'EMBEDDING_MODEL_NAME', 'N/A')}")
        st.write(f"**Dataset:** MIND (MINDsmall) — via `src/config.py` paths")
        st.write(f"**Number of embeddings loaded:** {n_embeddings}")
        st.write(f"**LightGBM model loaded:** {'Yes' if lightgbm_model is not None else 'No'}")
        st.write(f"**Logistic Regression model loaded:** {'Yes' if logreg_model is not None else 'No'}")
        st.write(f"**FAISS index loaded:** {'Yes' if index is not None else 'No'}")


# ---------------------------------------------------------------------------
# Build history_ids (shared by Recommendations tab)
# ---------------------------------------------------------------------------
history_ids = []

if mode == "Select existing user" and can_use_real_users:
    warm_users = (
        behaviors_df[behaviors_df[history_col].astype(str).str.len() > 0]
        .drop_duplicates(subset=user_col)[user_col]
    )
    if warm_users.empty:
        st.sidebar.warning("No users with reading history found.")
    else:
        sample_users = warm_users.sample(min(200, len(warm_users)), random_state=42).tolist()
        selected_user = st.sidebar.selectbox("Select a UserID", sample_users)
        user_row = behaviors_df[behaviors_df[user_col] == selected_user].iloc[0]
        history_str_raw = user_row[history_col] if isinstance(user_row[history_col], str) else ""
        history_ids = history_str_raw.split() if history_str_raw else []
else:
    category_col = NEWS_COLS.get("Category")
    if category_col:
        categories = sorted(news_df[category_col].dropna().unique().tolist())
        chosen = st.sidebar.multiselect("Topics of interest", categories, default=categories[:2] if categories else [])
        if chosen:
            pool = news_df[news_df[category_col].isin(chosen)]
            id_col = NEWS_COLS.get("NewsID")
            if id_col:
                history_ids = pool.sample(min(5, len(pool)), random_state=1)[id_col].tolist()
    else:
        st.sidebar.info("No category column detected — cannot simulate a new user by topic.")


# ---------------------------------------------------------------------------
# RECOMMENDATIONS TAB
# ---------------------------------------------------------------------------
with tab_recs:
    st.subheader("📖 Reading History")

    id_col = NEWS_COLS.get("NewsID")
    title_col = NEWS_COLS.get("Title")
    category_col = NEWS_COLS.get("Category")

    if history_ids and id_col:
        hist_rows = news_df[news_df[id_col].isin(history_ids)]
        chips_html = "".join(
            f'<span class="history-chip"><b>{get(r, NEWS_COLS, "Category", "—")}</b> · '
            f'{str(get(r, NEWS_COLS, "Title", ""))[:50]}'
            f'{"…" if len(str(get(r, NEWS_COLS, "Title", ""))) > 50 else ""}</span>'
            for r in hist_rows.itertuples()
        )
        st.markdown(chips_html, unsafe_allow_html=True)
    else:
        st.info("No reading history — this user will receive popularity-based cold-start recommendations.")

    st.markdown("---")
    st.subheader("✨ Recommended For You")

    recs = pd.DataFrame()
    score_label = "score"
    is_cold_start = True

    try:
        can_personalize = (
            history_ids and get_user_vector and news_id_to_idx is not None and news_embeddings is not None
        )
        user_vector = (
            get_user_vector(" ".join(history_ids), news_id_to_idx, news_embeddings)
            if can_personalize else None
        )
        is_cold_start = user_vector is None

        if is_cold_start:
            if popularity_df is not None and id_col:
                recs = (
                    news_df.merge(popularity_df, on=id_col, how="left")
                    if "NewsID" in popularity_df.columns and id_col == "NewsID"
                    else news_df.copy()
                )
                if "popularity_score" in recs.columns:
                    recs["popularity_score"] = recs["popularity_score"].fillna(0.0)
                    recs = recs.sort_values("popularity_score", ascending=False)
                    recs["final_score"] = recs["popularity_score"]
                    score_label = "popularity score"
                else:
                    recs = recs.sample(min(top_k, len(recs)), random_state=0)
                    recs["final_score"] = 0.0
                    score_label = "random (no popularity data)"
                recs = recs.head(top_k).reset_index(drop=True)
            else:
                recs = news_df.sample(min(top_k, len(news_df)), random_state=0).reset_index(drop=True)
                recs["final_score"] = 0.0
                score_label = "random (no popularity data)"
            st.warning("Cold-start user — showing trending/sample articles instead of personalized ones.")
        elif recommend_candidates and index is not None:
            candidates = recommend_candidates(user_vector, index, news_df, top_k=config.CANDIDATE_POOL_SIZE)

            if strategy.startswith("Similarity") and rank_candidates_heuristic and popularity_df is not None:
                recs = rank_candidates_heuristic(candidates, popularity_df, top_k=top_k)
                score_label = "hybrid score"
            elif strategy == "Logistic Regression" and rank_candidates_ml:
                recs = rank_candidates_ml(
                    candidates, history_ids, news_df, news_id_to_idx, news_embeddings,
                    popularity_df, logreg_model, is_lightgbm=False, top_k=top_k,
                )
                score_label = "click probability"
            elif strategy == "LightGBM" and rank_candidates_ml:
                recs = rank_candidates_ml(
                    candidates, history_ids, news_df, news_id_to_idx, news_embeddings,
                    popularity_df, lightgbm_model, is_lightgbm=True, top_k=top_k,
                )
                score_label = "click probability"
            else:
                recs = candidates.head(top_k).reset_index(drop=True)
                recs["final_score"] = recs.get("similarity_score", 0.0)
                score_label = "similarity score"
        else:
            st.warning("FAISS index or candidate retrieval unavailable — cannot personalize. Showing sample articles.")
            recs = news_df.sample(min(top_k, len(news_df)), random_state=0).reset_index(drop=True)
            recs["final_score"] = 0.0
            score_label = "random"

    except Exception as e:
        st.error(f"Failed to generate recommendations: {e}")
        st.code(traceback.format_exc())
        recs = pd.DataFrame()

    # --- Metrics row ---
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (len(history_ids), "History Articles"),
        (len(recs), "Recommendations"),
        (strategy.split(" ")[0], "Ranking Model"),
        (getattr(config, "CANDIDATE_POOL_SIZE", "—") if not is_cold_start else "—", "Candidate Pool"),
    ]
    for col, (value, label) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(
                f'<div class="metric-box"><div class="metric-num">{value}</div>'
                f'<div class="metric-lbl">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Recommendation cards ---
    if recs.empty:
        st.info("No recommendations could be generated for this user/settings combination.")
    else:
        left, right = st.columns(2)

        for i, row in enumerate(recs.itertuples()):
            title = get(row, NEWS_COLS, "Title", "Untitled")
            category = get(row, NEWS_COLS, "Category", "—")
            abstract = get(row, NEWS_COLS, "Abstract", "")
            news_id_val = get(row, NEWS_COLS, "NewsID", None)

            final_score = getattr(row, "final_score", None)
            similarity_score = getattr(row, "similarity_score", None)
            popularity_score = getattr(row, "popularity_score", None)

            # Explainability
            explanation = None
            similarity_pct = None
            if not is_cold_start and history_ids and most_similar_history_article and news_id_val is not None:
                try:
                    similar_id = most_similar_history_article(
                        news_id_val, history_ids, news_id_to_idx, news_embeddings
                    )
                    if similar_id and id_col:
                        match = news_df.loc[news_df[id_col] == similar_id]
                        if not match.empty:
                            match_title = get(match.iloc[0], NEWS_COLS, "Title", "")
                            explanation = f'Recommended because you previously read: "{str(match_title)[:60]}"'
                            if similarity_score is not None:
                                similarity_pct = similarity_score * 100
                except Exception:
                    pass

            # Score chips — only show whichever scores actually exist
            chips = []
            if final_score is not None:
                chips.append(f'<span class="score-chip">🎯 {score_label}: {final_score:.3f}</span>')
            if similarity_score is not None:
                chips.append(f'<span class="score-chip blue">📐 similarity: {similarity_score:.3f}</span>')
            if popularity_score is not None:
                chips.append(f'<span class="score-chip amber">🔥 popularity: {popularity_score:.3f}</span>')
            if strategy in ("Logistic Regression", "LightGBM") and final_score is not None:
                chips.append(f'<span class="score-chip purple">🤖 ML probability: {final_score:.3f}</span>')

            score_row_html = f'<div class="score-row">{"".join(chips)}</div>' if chips else ""

            explain_text = explanation or ""
            if similarity_pct is not None:
                explain_text += f" (Similarity = {similarity_pct:.2f}%)"
            explain_html = f'<div class="rec-explain">💡 {explain_text}</div>' if explain_text else ""

            abstract_html = str(abstract) if isinstance(abstract, str) and abstract.strip() else "No abstract available."

            card_html = f"""
            <div class="rec-card">
                <span class="rec-rank">#{i + 1}</span>
                <span class="rec-category">{category}</span>
                <div class="rec-title">{title}</div>
                <div class="rec-abstract">{abstract_html}</div>
                {score_row_html}
                {explain_html}
            </div>
            """

            target = left if i % 2 == 0 else right
            with target:
                st.markdown(card_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# ARCHITECTURE TAB
# ---------------------------------------------------------------------------
with tab_arch:
    st.subheader("🧩 Recommendation Pipeline")

    pipeline_steps = [
        "📂 Dataset (MIND: news.tsv + behaviors.tsv)",
        "🧹 Preprocessing (clean text, build unified `text` field)",
        "🧠 Sentence Transformer (MiniLM embedding model)",
        "🔢 Embeddings (dense vector representations)",
        "⚡ FAISS Candidate Retrieval (cosine similarity search)",
        "🛠️ Feature Engineering (similarity, popularity, affinity, etc.)",
        "🤖 Logistic Regression / LightGBM Ranking",
        "✅ Final Recommendation",
    ]

    for i, step in enumerate(pipeline_steps):
        st.markdown(f'<div class="pipeline-step">{step}</div>', unsafe_allow_html=True)
        if i < len(pipeline_steps) - 1:
            st.markdown('<p class="pipeline-arrow">↓</p>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🗂️ Module Map")
    st.markdown(
        """
        | Stage | Module |
        |---|---|
        | Config | `src/config.py` |
        | Preprocessing | `src/preprocess.py` |
        | Embeddings | `src/embedding.py` |
        | FAISS index | `src/faiss_index.py` |
        | Candidate retrieval | `src/recommender.py` |
        | Feature engineering | `src/features.py` |
        | User profiles | `src/user_profile.py` |
        | Ranking (heuristic + ML) | `src/ranking.py`, `src/ranker.py` |
        | Explainability | `src/explainability.py` |
        | Evaluation | `src/evaluation.py` |
        """
    )


# ---------------------------------------------------------------------------
# ABOUT TAB
# ---------------------------------------------------------------------------
with tab_about:
    st.subheader("ℹ️ About This Project")
    st.markdown(
        """
        This is a **Personalized News Recommendation System** built on Microsoft's
        MIND (MIcrosoft News Dataset) dataset. It demonstrates a full two-stage
        recommendation pipeline:

        1. **Retrieval** — Sentence-BERT embeddings + FAISS for fast candidate generation.
        2. **Ranking** — A learned click-through model (Logistic Regression / LightGBM)
           re-ranks candidates using engineered features (similarity, popularity,
           category affinity, and more).
        3. **Explainability** — Every recommendation can be traced back to the most
           similar article in the user's reading history.

        This Streamlit app is a read-only demo: it loads artifacts already produced by
        the project's notebooks (`01_Exploration.ipynb`, `02_Model_Development.ipynb`,
        `03_Evaluation.ipynb`) and does not train anything live.
        """
    )
    st.markdown("---")
    st.caption(
        "Dataset: Microsoft MIND (MINDsmall) · Stack: Python, sentence-transformers, "
        "FAISS, LightGBM, scikit-learn, Streamlit"
    )