"""
Streamlit App - Movie Recommendation System
Interactive UI with dark-theme, animated cards, and multiple features.
"""

import os
import time
import pickle
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from io import BytesIO

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 CineMatch – Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
:root {
    --gold: #f5c518;
    --red:  #e50914;
    --dark: #141414;
    --card: #1f1f1f;
    --text: #e5e5e5;
}
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--dark) !important;
    color: var(--text);
    font-family: 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #0d0d0d !important;
    border-right: 1px solid #2a2a2a;
}
/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
    border-radius: 16px;
    padding: 40px 50px;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(229, 9, 20, 0.3);
}
.hero::before {
    content: "";
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(229,9,20,0.08) 0%, transparent 60%);
    animation: pulse 4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f5c518, #e50914);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.hero p {
    font-size: 1.1rem;
    color: #aaaaaa;
}
/* ── Movie Cards ── */
.movie-card {
    background: var(--card);
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    border: 1px solid #2a2a2a;
    transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
    cursor: pointer;
    position: relative;
    overflow: hidden;
}
.movie-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0;
    width: 4px;
    height: 100%;
    background: linear-gradient(180deg, #f5c518, #e50914);
    border-radius: 12px 0 0 12px;
}
.movie-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(229, 9, 20, 0.25);
    border-color: #e50914;
}
.movie-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 6px;
}
.movie-meta {
    font-size: 0.82rem;
    color: #888;
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 8px;
}
.badge {
    background: rgba(245, 197, 24, 0.15);
    border: 1px solid rgba(245, 197, 24, 0.4);
    color: #f5c518;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-red {
    background: rgba(229, 9, 20, 0.15);
    border: 1px solid rgba(229, 9, 20, 0.4);
    color: #ff6b6b;
}
.badge-blue {
    background: rgba(80, 160, 255, 0.15);
    border: 1px solid rgba(80, 160, 255, 0.4);
    color: #7eb8ff;
}
.sim-bar {
    height: 6px;
    border-radius: 3px;
    background: linear-gradient(90deg, #e50914, #f5c518);
    margin-top: 6px;
}
/* ── Metric Cards ── */
.stat-card {
    background: linear-gradient(135deg, var(--card), #2a1a1a);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    border: 1px solid #3a2a2a;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}
.stat-num {
    font-size: 2rem;
    font-weight: 800;
    color: var(--gold);
}
.stat-label {
    font-size: 0.85rem;
    color: #888;
    margin-top: 4px;
}
/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 30px 0 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid #2a2a2a;
}
.section-header h2 {
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
}
/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #e50914, #c0000e) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 12px rgba(229, 9, 20, 0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(229, 9, 20, 0.6) !important;
}
.stSelectbox > div > div {
    background-color: #2a2a2a !important;
    border-color: #3a3a3a !important;
    color: var(--text) !important;
}
.stSlider > div { color: var(--text) !important; }
div[data-testid="stMetric"] {
    background-color: #1f1f1f;
    border-radius: 10px;
    padding: 12px;
    border: 1px solid #2a2a2a;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #888 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important;
}
/* ── Search box ── */
.stTextInput > div > div > input {
    background-color: #2a2a2a !important;
    color: white !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 8px !important;
}
div[data-testid="stExpander"] {
    background-color: #1f1f1f;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
TMDB_IMG = "https://image.tmdb.org/t/p/w300"
PLACEHOLDER = "https://via.placeholder.com/300x450/1f1f1f/888888?text=No+Poster"

@st.cache_resource(show_spinner=False)
def load_recommender():
    """Load (or train) the recommender model."""
    model_path = "models/recommender.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            rec = pickle.load(f)
        return rec, None

    # Train on the fly
    try:
        from recommender import train
        with st.spinner("🔧 First-time setup: training model (this takes ~30s)..."):
            rec = train()
        return rec, None
    except Exception as e:
        return None, str(e)


def star_rating(score: float) -> str:
    filled = int(round(score / 2))
    return "⭐" * filled + "☆" * (5 - filled)


def fmt_runtime(mins):
    try:
        m = int(mins)
        return f"{m // 60}h {m % 60}m" if m > 0 else "N/A"
    except Exception:
        return "N/A"


def poster_url(path):
    if isinstance(path, str) and path.startswith("/"):
        return TMDB_IMG + path
    return PLACEHOLDER


def render_movie_card(row, show_sim=False, sim_val=None):
    """Render a single movie card."""
    genres_html = "".join(
        f'<span class="badge">{g.strip()}</span>'
        for g in str(row.get("genres", row.get("genres_str", ""))).split(",")[:3]
        if g.strip()
    )
    year = int(row.get("year", 0))
    rating = float(row.get("vote_average", 0))
    sim_pct = int((sim_val or 0) * 100)
    sim_bar = (
        f'<div style="display:flex;align-items:center;gap:8px;margin-top:8px;">'
        f'<span style="font-size:0.8rem;color:#888;">Match</span>'
        f'<div style="flex:1;height:6px;border-radius:3px;background:#2a2a2a;">'
        f'<div class="sim-bar" style="width:{sim_pct}%"></div></div>'
        f'<span style="font-size:0.8rem;color:#f5c518;font-weight:600;">{sim_pct}%</span>'
        f'</div>'
    ) if show_sim and sim_val else ""

    overview = str(row.get("overview", ""))[:160]
    overview += "..." if len(str(row.get("overview", ""))) > 160 else ""

    html = f"""
<div class="movie-card">
    <div class="movie-title">{row.get('title','?')}</div>
    <div class="movie-meta">
        <span class="badge-red badge">⭐ {rating:.1f}</span>
        <span>📅 {year if year else 'N/A'}</span>
        <span>⏱ {fmt_runtime(row.get('runtime', 0))}</span>
    </div>
    <div style="margin-bottom:8px;">{genres_html}</div>
    <div style="font-size:0.82rem;color:#aaa;line-height:1.5;">{overview}</div>
    {sim_bar}
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def render_top_movie_card(row):
    """Render card for top-rated movies list."""
    genres_html = "".join(
        f'<span class="badge">{g.strip()}</span>'
        for g in str(row.get("genres_str", "")).split(",")[:3]
        if g.strip()
    )
    year = int(row.get("year", 0)) if row.get("year") else 0
    rating = float(row.get("vote_average", 0))
    overview = str(row.get("overview", ""))[:140]
    if len(str(row.get("overview", ""))) > 140:
        overview += "..."

    html = f"""
<div class="movie-card">
    <div class="movie-title">{row.get('title','?')}</div>
    <div class="movie-meta">
        <span class="badge-red badge">⭐ {rating:.1f}</span>
        <span>📅 {year if year else 'N/A'}</span>
        <span>🎬 {row.get('director','')}</span>
    </div>
    <div style="margin-bottom:8px;">{genres_html}</div>
    <div style="font-size:0.82rem;color:#aaa;line-height:1.5;">{overview}</div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────
def main():
    # ── Load Model ──
    rec, err = load_recommender()

    if err:
        st.error(f"❌ Could not load the recommendation model:\n\n`{err}`\n\n"
                 "Make sure the dataset files are in the `data/` folder and run `python recommender.py` first.")
        st.stop()

    df = rec.df
    all_titles = sorted(df["title"].dropna().unique().tolist())
    all_genres = sorted(set(
        g for genres in df["genres_list"] for g in genres
    ))

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:20px 0;">
            <div style="font-size:3rem;">🎬</div>
            <div style="font-size:1.4rem;font-weight:800;
                        background:linear-gradient(90deg,#f5c518,#e50914);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                CineMatch
            </div>
            <div style="color:#666;font-size:0.8rem;margin-top:4px;">
                AI Movie Recommender
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        page = st.radio(
            "Navigation",
            ["🏠 Home", "🔍 Get Recommendations", "🏆 Top Movies",
             "📊 Analytics", "🔎 Search Movies"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.75rem;color:#555;text-align:center;padding:10px 0;">
            Built with ❤️ using<br>Streamlit + Scikit-Learn<br>
            Dataset: TMDB 5000 Movies
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────
    # PAGE: HOME
    # ─────────────────────────────────────────────────────────────
    if page == "🏠 Home":
        st.markdown("""
        <div class="hero">
            <h1>🎬 CineMatch</h1>
            <p>Your AI-powered movie companion. Discover films tailored to your taste using<br>
            <strong style="color:#f5c518;">content-based machine learning</strong> with TF-IDF & cosine similarity.</p>
        </div>
        """, unsafe_allow_html=True)

        # Stats row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-num">{len(df):,}</div>
                <div class="stat-label">🎬 Movies</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            avg = df["vote_average"].mean()
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-num">{avg:.1f}</div>
                <div class="stat-label">⭐ Avg Rating</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-num">{len(all_genres)}</div>
                <div class="stat-label">🎭 Genres</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            years = df["year"][df["year"] > 1900]
            span = f"{int(years.min())}–{int(years.max())}" if len(years) > 0 else "N/A"
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-num" style="font-size:1.4rem;">{span}</div>
                <div class="stat-label">📅 Year Span</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Quick pick
        st.markdown('<div class="section-header"><h2>⚡ Quick Recommend</h2></div>', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])
        with col1:
            quick_title = st.selectbox("Pick a movie you like:", all_titles, key="home_pick")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            go_btn = st.button("🚀 Get Recommendations", use_container_width=True)

        if go_btn and quick_title:
            recs = rec.recommend(quick_title, n=6)
            if recs.empty:
                st.warning("No recommendations found.")
            else:
                cols = st.columns(2)
                for i, (_, row) in enumerate(recs.iterrows()):
                    with cols[i % 2]:
                        render_movie_card(row, show_sim=True, sim_val=row["similarity"])

        # Trending section
        st.markdown('<div class="section-header"><h2>🔥 Trending Now</h2></div>', unsafe_allow_html=True)
        trending = df.sort_values("popularity", ascending=False).head(6)
        cols = st.columns(3)
        for i, (_, row) in enumerate(trending.iterrows()):
            with cols[i % 3]:
                render_top_movie_card(row)

    # ─────────────────────────────────────────────────────────────
    # PAGE: GET RECOMMENDATIONS
    # ─────────────────────────────────────────────────────────────
    elif page == "🔍 Get Recommendations":
        st.markdown("""
        <div class="section-header">
            <h2>🔍 Personalized Recommendations</h2>
        </div>""", unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.markdown("#### 🎯 Configure")
            selected_movie = st.selectbox("Select a movie you love:", all_titles)
            n_recs = st.slider("Number of recommendations:", 5, 20, 10)
            min_rating = st.slider("Minimum IMDb rating:", 0.0, 9.0, 5.0, 0.5)
            genre_filter = st.multiselect("Filter by genre (optional):", all_genres)
            year_min, year_max = st.slider(
                "Year range:", 1950, 2020, (1990, 2020)
            )
            recommend_btn = st.button("🎬 Find Similar Movies", use_container_width=True)

            # Show selected movie info
            if selected_movie:
                movie_info = rec.get_movie_info(selected_movie)
                if movie_info is not None:
                    st.markdown("---")
                    st.markdown("#### 📋 Selected Movie")
                    st.markdown(f"""
                    <div class="movie-card" style="border-color:#f5c518;">
                        <div class="movie-title">{movie_info['title']}</div>
                        <div class="movie-meta">
                            <span class="badge-red badge">⭐ {movie_info['vote_average']:.1f}</span>
                            <span>📅 {int(movie_info['year']) if movie_info['year'] else 'N/A'}</span>
                        </div>
                        <div style="font-size:0.82rem;color:#aaa;">{str(movie_info.get('overview',''))[:200]}</div>
                    </div>
                    """, unsafe_allow_html=True)

        with col_right:
            if recommend_btn and selected_movie:
                with st.spinner("🤖 Finding your perfect matches..."):
                    time.sleep(0.3)
                    recs = rec.recommend(
                        selected_movie,
                        n=n_recs,
                        genre_filter=genre_filter if genre_filter else None,
                        year_range=(year_min, year_max),
                        min_rating=min_rating,
                    )

                if recs.empty:
                    st.warning("😕 No movies match your filters. Try relaxing the constraints!")
                else:
                    st.markdown(f"#### ✅ Top {len(recs)} Matches for *{selected_movie}*")

                    # Summary chart
                    fig = px.bar(
                        recs.head(10),
                        x="similarity",
                        y="title",
                        orientation="h",
                        color="vote_average",
                        color_continuous_scale=["#1f1f1f", "#e50914", "#f5c518"],
                        title="Similarity Scores",
                        labels={"similarity": "Match %", "title": "",
                                "vote_average": "Rating"},
                    )
                    fig.update_layout(
                        paper_bgcolor="#141414",
                        plot_bgcolor="#1f1f1f",
                        font_color="#e5e5e5",
                        title_font_size=14,
                        height=300,
                        margin=dict(l=0, r=0, t=40, b=0),
                        coloraxis_colorbar=dict(tickfont=dict(color="#e5e5e5")),
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Cards
                    cols = st.columns(2)
                    for i, (_, row) in enumerate(recs.iterrows()):
                        with cols[i % 2]:
                            render_movie_card(row, show_sim=True, sim_val=row["similarity"])

            elif not recommend_btn:
                st.markdown("""
                <div style="display:flex;flex-direction:column;align-items:center;
                            justify-content:center;height:400px;color:#444;">
                    <div style="font-size:4rem;">🎬</div>
                    <div style="font-size:1.1rem;margin-top:16px;">
                        Select a movie and click <strong style="color:#f5c518;">Find Similar Movies</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────
    # PAGE: TOP MOVIES
    # ─────────────────────────────────────────────────────────────
    elif page == "🏆 Top Movies":
        st.markdown('<div class="section-header"><h2>🏆 Top Rated Movies</h2></div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            genre_pick = st.selectbox("Genre:", ["All"] + all_genres)
        with c2:
            top_year_min, top_year_max = st.slider("Year range:", 1950, 2020, (2000, 2020), key="top_yr")
        with c3:
            min_top_rating = st.slider("Min rating:", 5.0, 9.5, 7.0, 0.5, key="top_r")

        top = rec.top_movies(
            genre=genre_pick if genre_pick != "All" else None,
            year_range=(top_year_min, top_year_max),
            min_rating=min_top_rating,
            n=24,
        )

        if top.empty:
            st.warning("No movies found with these filters.")
        else:
            st.markdown(f"*Showing **{len(top)}** movies*")
            cols = st.columns(3)
            for i, (_, row) in enumerate(top.iterrows()):
                with cols[i % 3]:
                    render_top_movie_card(row)

    # ─────────────────────────────────────────────────────────────
    # PAGE: ANALYTICS
    # ─────────────────────────────────────────────────────────────
    elif page == "📊 Analytics":
        st.markdown('<div class="section-header"><h2>📊 Dataset Analytics</h2></div>', unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs([
            "🎭 Genre Insights", "📈 Rating Distribution",
            "📅 Movies Over Time", "💰 Budget vs Revenue"
        ])

        chart_theme = dict(paper_bgcolor="#141414", plot_bgcolor="#1f1f1f",
                           font_color="#e5e5e5")

        with tab1:
            genre_df = rec.genre_analysis()
            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    genre_df.head(15), x="count", y="genre",
                    orientation="h", title="Movies per Genre",
                    color="avg_rating",
                    color_continuous_scale=["#1f1f1f", "#e50914", "#f5c518"],
                    labels={"count": "# Movies", "genre": "", "avg_rating": "Avg Rating"},
                )
                fig.update_layout(**chart_theme, height=450,
                                  yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.scatter(
                    genre_df, x="avg_rating", y="avg_popularity",
                    size="count", color="genre",
                    title="Genre: Rating vs Popularity",
                    labels={"avg_rating": "Avg Rating",
                            "avg_popularity": "Avg Popularity"},
                    hover_name="genre",
                )
                fig.update_layout(**chart_theme, height=450, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig = px.histogram(
                df[df["vote_average"] > 0], x="vote_average",
                nbins=40, title="Rating Distribution",
                color_discrete_sequence=["#e50914"],
                labels={"vote_average": "IMDb Rating", "count": "# Movies"},
            )
            fig.update_layout(**chart_theme, bargap=0.05)
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.box(
                df[df["vote_average"] > 0], x="vote_average",
                title="Rating Box Plot",
                color_discrete_sequence=["#f5c518"],
            )
            fig2.update_layout(**chart_theme)
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            yearly = (
                df[df["year"] > 1950]
                .groupby("year")
                .agg(count=("title", "count"),
                     avg_rating=("vote_average", "mean"))
                .reset_index()
            )
            fig = px.line(
                yearly, x="year", y="count",
                title="Movies Released per Year",
                labels={"year": "Year", "count": "# Movies"},
                color_discrete_sequence=["#e50914"],
            )
            fig.update_traces(fill="tonexty", fillcolor="rgba(229,9,20,0.1)")
            fig.update_layout(**chart_theme)
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.line(
                yearly, x="year", y="avg_rating",
                title="Average Rating Over Years",
                labels={"year": "Year", "avg_rating": "Avg Rating"},
                color_discrete_sequence=["#f5c518"],
            )
            fig2.update_layout(**chart_theme)
            st.plotly_chart(fig2, use_container_width=True)

        with tab4:
            budget_df = df[(df["budget"] > 1e6) & (df["revenue"] > 1e6)].copy()
            budget_df["roi"] = budget_df["revenue"] / budget_df["budget"]
            fig = px.scatter(
                budget_df, x="budget", y="revenue",
                color="vote_average",
                size="popularity",
                hover_name="title",
                title="Budget vs Revenue",
                color_continuous_scale=["#1f1f1f", "#e50914", "#f5c518"],
                log_x=True, log_y=True,
                labels={"budget": "Budget (\$)", "revenue": "Revenue (\$)"},
                opacity=0.7,
            )
            fig.update_layout(**chart_theme, height=500)
            st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────────────────────────
    # PAGE: SEARCH
    # ─────────────────────────────────────────────────────────────
    elif page == "🔎 Search Movies":
        st.markdown('<div class="section-header"><h2>🔎 Search Movies</h2></div>', unsafe_allow_html=True)

        query = st.text_input("🔍 Search by title:", placeholder="e.g. Inception, Dark Knight...")

        if query and len(query) >= 2:
            results = rec.search(query, n=20)
            if results.empty:
                st.warning(f"No movies found for '{query}'")
            else:
                st.markdown(f"*Found **{len(results)}** results for **'{query}'***")
                cols = st.columns(2)
                for i, (_, row) in enumerate(results.iterrows()):
                    with cols[i % 2]:
                        genres_html = "".join(
                            f'<span class="badge">{g.strip()}</span>'
                            for g in str(row.get("genres_str", "")).split(",")[:3]
                            if g.strip()
                        )
                        html = f"""
<div class="movie-card">
    <div class="movie-title">{row.get('title','?')}</div>
    <div class="movie-meta">
        <span class="badge-red badge">⭐ {float(row.get('vote_average', 0)):.1f}</span>
        <span>📅 {int(row.get('year', 0)) if row.get('year') else 'N/A'}</span>
    </div>
    <div style="margin-bottom:6px;">{genres_html}</div>
    <div style="font-size:0.82rem;color:#aaa;">{str(row.get('overview',''))[:180]}</div>
</div>"""
                        st.markdown(html, unsafe_allow_html=True)

                        # "Get Recs" button
                        if st.button(f"🎬 Recs for this", key=f"srec_{i}"):
                            recs = rec.recommend(row["title"], n=6)
                            if not recs.empty:
                                st.markdown(f"**Similar to *{row['title']}*:**")
                                for _, rrow in recs.iterrows():
                                    render_movie_card(rrow, show_sim=True, sim_val=rrow["similarity"])
        else:
            st.markdown("""
            <div style="text-align:center;padding:60px 0;color:#444;">
                <div style="font-size:3rem;">🔍</div>
                <div style="margin-top:12px;">Type at least 2 characters to search...</div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
