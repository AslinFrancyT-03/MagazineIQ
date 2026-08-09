"""
MagazineIQ Dashboard
====================
Streamlit frontend integrating all eight backend NLP modules into a
production-ready, multi-page analytics dashboard.

Backend modules:
    PDFExtractor, ArticleDetector, KeywordExtractor, EntityExtractor,
    Summarizer, TopicClassifier, SentimentAnalyzer, AnalyticsEngine
"""

import json
import os
import logging
import tempfile
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.text_extraction import PDFExtractor
from src.preprocessing import ArticleDetector
from src.feature_engineering import KeywordExtractor
from src.ner import EntityExtractor
from src.summarizer import Summarizer
from src.topic_modeling import TopicClassifier
from src.sentiment import SentimentAnalyzer
from src.analytics import AnalyticsEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENTIMENT_COLORS: Dict[str, str] = {
    "POSITIVE": "#A36A2A",
    "NEUTRAL":  "#8B6F47",
    "NEGATIVE": "#3B2416",
}

SENTIMENT_ICONS: Dict[str, str] = {
    "POSITIVE": "✦",
    "NEUTRAL":  "❖",
    "NEGATIVE": "✧",
}

TOPIC_COLORS: List[str] = [
    "#5A3E2B", "#8B6F47", "#A36A2A", "#6B5D45", "#A88454",
    "#C6A675", "#D8B97B", "#3B2416", "#7D6246", "#9E8262",
    "#4A3F35", "#E8DCC8",
]

# Shared Plotly layout for all charts — vintage parchment aesthetic
_CHART_LAYOUT = dict(
    paper_bgcolor="#F4E6C1",
    plot_bgcolor="#F4E6C1",
    font=dict(family="EB Garamond, Georgia, serif", color="#3B2416", size=12),
    title_font=dict(family="Cormorant Garamond, Georgia, serif", color="#3B2416", size=18),
    xaxis=dict(
        gridcolor="rgba(168,132,84,0.2)",
        linecolor="#A88454",
        tickfont=dict(family="EB Garamond, Georgia, serif", color="#3B2416"),
        title_font=dict(family="EB Garamond, Georgia, serif", color="#3B2416"),
    ),
    yaxis=dict(
        gridcolor="rgba(168,132,84,0.2)",
        linecolor="#A88454",
        tickfont=dict(family="EB Garamond, Georgia, serif", color="#3B2416"),
        title_font=dict(family="EB Garamond, Georgia, serif", color="#3B2416"),
    ),
    legend=dict(
        font=dict(family="Cinzel, Georgia, serif", color="#3B2416", size=11),
        bgcolor="rgba(244,230,193,0.8)",
        bordercolor="#A88454",
        borderwidth=1,
    ),
    margin=dict(t=36, b=28, l=28, r=28),
)

MAX_UPLOAD_MB: int = 200


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _model_settings() -> Dict[str, str]:
    """Current model names from session state with safe defaults.
    Sanitises stale session values that may reference deprecated models."""
    _VALID_SUMMARIZER = "Falconsai/text_summarization"
    _VALID_CLASSIFIER = "typeform/distilbert-base-uncased-mnli"

    # Force-correct any stale session state from previous runs
    cur = st.session_state.get("cfg_summarizer", _VALID_SUMMARIZER)

    cur_cls = st.session_state.get("cfg_classifier", _VALID_CLASSIFIER)
    if "bart-large-mnli" in cur_cls.lower():
        st.session_state["cfg_classifier"] = _VALID_CLASSIFIER
        cur_cls = _VALID_CLASSIFIER

    return {
        "summarizer": cur,
        "classifier": cur_cls,
        "sentiment": st.session_state.get(
            "cfg_sentiment", "distilbert-base-uncased-finetuned-sst-2-english"
        ),
    }


def _has_data() -> bool:
    return "magazine_data" in st.session_state and st.session_state["magazine_data"] is not None


def _data() -> Dict[str, Any]:
    return st.session_state["magazine_data"]


# ---------------------------------------------------------------------------
# NLP Pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(
    file_path: str,
    callback: Callable[[str], None],
    progress_bar: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """
    Runs the eight-stage NLP pipeline and returns the combined result dict.

    Stages:
        1. PDFExtractor.extract()
        2. ArticleDetector(pages).detect()
        3. KeywordExtractor().extract_from_articles() + aggregate_magazine_keywords()
        4. EntityExtractor().extract_from_articles() + aggregate_magazine_entities()
        5. Summarizer().summarize_articles() + generate_magazine_summary()
        6. TopicClassifier().classify_articles() + aggregate_magazine_topics()
        7. SentimentAnalyzer().analyze_articles() + aggregate_magazine_sentiment()
        8. AnalyticsEngine(articles, metadata).generate_report()
    """
    models = _model_settings()
    total = 8
    stage = 0

    def advance(msg: str) -> None:
        nonlocal stage
        stage += 1
        callback(msg)
        if progress_bar is not None:
            progress_bar.progress(stage / total, text=f"Stage {stage}/{total}: {msg}")

    try:
        # 1 ── PDF Extraction
        advance("📄 Extracting text from PDF...")
        extractor = PDFExtractor(file_path)
        extraction = extractor.extract()
        if extraction["status"] != "success":
            st.error(f"PDF extraction failed: {extraction['error_message']}")
            return None

        pages = extraction["pages"]
        metadata = extraction["metadata"]

        # 2 ── Article Detection
        advance("🔍 Detecting article boundaries...")
        detector = ArticleDetector(pages)
        detection = detector.detect()
        articles: List[Dict[str, Any]] = detection["articles"]
        if not articles:
            st.warning("No articles detected in this magazine.")
            return None

        # 3 ── Keyword Extraction
        advance("🔑 Extracting keywords (YAKE)...")
        kw_ext = KeywordExtractor()
        articles = kw_ext.extract_from_articles(articles)
        mag_keywords = kw_ext.aggregate_magazine_keywords(articles)

        # 4 ── Entity Extraction
        advance("🏷️ Extracting named entities (spaCy)...")
        ent_ext = EntityExtractor()
        articles = ent_ext.extract_from_articles(articles)
        mag_entities = ent_ext.aggregate_magazine_entities(articles)

        # 5 ── Summarisation
        advance("📝 Generating summaries (this may take several minutes)...")
        summarizer = Summarizer(
            model_name=models["summarizer"],
            progress_callback=callback,
        )
        articles = summarizer.summarize_articles(articles)
        mag_summary = summarizer.generate_magazine_summary(articles)

        # 6 ── Topic Classification
        advance("📂 Classifying topics (zero-shot)...")
        classifier = TopicClassifier(model_name=models["classifier"])
        articles = classifier.classify_articles(articles)
        mag_topics = classifier.aggregate_magazine_topics(articles)

        # 7 ── Sentiment Analysis
        advance("💬 Analysing sentiment...")
        sentiment = SentimentAnalyzer(model_name=models["sentiment"])
        articles = sentiment.analyze_articles(articles)
        mag_sentiment = sentiment.aggregate_magazine_sentiment(articles)

        # 8 ── Analytics Report
        advance("📊 Computing analytics report...")
        engine = AnalyticsEngine(articles, metadata)
        report = engine.generate_report()

        return {
            "metadata": metadata,
            "articles": articles,
            "magazine_summary": mag_summary,
            "magazine_keywords": mag_keywords,
            "magazine_entities": mag_entities,
            "magazine_topics": mag_topics,
            "magazine_sentiment": mag_sentiment,
            "article_statistics": detection["statistics"],
            "report": report,
            "processed_at": datetime.now().isoformat(),
            "models_used": models,
        }

    except Exception as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True)
        st.error(f"An error occurred during analysis: {exc}")
        return None


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _apply_vintage_layout(fig: go.Figure, **overrides) -> go.Figure:
    """Apply the shared vintage parchment Plotly layout to any figure."""
    layout = dict(_CHART_LAYOUT)
    layout.update(overrides)
    fig.update_layout(**layout)
    return fig


def _pie_topics(dist: Dict[str, float]) -> go.Figure:
    fig = px.pie(
        names=list(dist.keys()),
        values=list(dist.values()),
        color_discrete_sequence=TOPIC_COLORS,
        hole=0.38,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        textfont=dict(family="EB Garamond, Georgia, serif", color="#F4E6C1", size=11),
        marker=dict(line=dict(color="#F4E6C1", width=2)),
    )
    _apply_vintage_layout(fig, showlegend=False)
    return fig


def _bar_sentiment(sent: Dict[str, Any]) -> go.Figure:
    labels = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
    vals = [
        sent.get("positive_percent", 0),
        sent.get("neutral_percent", 0),
        sent.get("negative_percent", 0),
    ]
    fig = go.Figure(go.Bar(
        x=labels,
        y=vals,
        marker_color=[SENTIMENT_COLORS[l] for l in labels],
        marker_line_color="#A88454",
        marker_line_width=1,
        text=[f"{v:.1f}%" for v in vals],
        textposition="auto",
        textfont=dict(family="Cinzel, Georgia, serif", color="#F8F2E5", size=11),
    ))
    _apply_vintage_layout(fig, yaxis_title="Percentage (%)", showlegend=False)
    return fig


def _bar_keywords(kw_list: List[Dict[str, Any]], n: int = 15) -> go.Figure:
    df = pd.DataFrame(kw_list[:n])
    if df.empty:
        return go.Figure()
    fig = px.bar(
        df, x="frequency", y="keyword", orientation="h",
        color="frequency",
        color_continuous_scale=[
            [0.0, "#E8DCC8"], [0.4, "#A88454"], [0.7, "#A36A2A"], [1.0, "#5A3E2B"]
        ],
    )
    fig.update_traces(marker_line_color="#A88454", marker_line_width=0.5)
    _apply_vintage_layout(
        fig,
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
    )
    return fig


def _bar_reading_time(articles: List[Dict[str, Any]]) -> go.Figure:
    rows = [
        {
            "Article": a.get("title", "Untitled")[:45],
            "Minutes": round(a.get("reading_time", 0) / 60, 1),
        }
        for a in articles
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return go.Figure()
    fig = px.bar(
        df.sort_values("Minutes", ascending=False).head(15),
        x="Minutes", y="Article", orientation="h",
        color="Minutes",
        color_continuous_scale=[
            [0.0, "#E8DCC8"], [0.4, "#A88454"], [0.7, "#8B6F47"], [1.0, "#3B2416"]
        ],
    )
    fig.update_traces(marker_line_color="#A88454", marker_line_width=0.5)
    _apply_vintage_layout(
        fig,
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
    )
    return fig


def _hist_word_counts(articles: List[Dict[str, Any]]) -> go.Figure:
    wc = [a.get("word_count", 0) for a in articles]
    fig = go.Figure(go.Histogram(
        x=wc,
        nbinsx=max(5, len(articles) // 2),
        marker_color="#8B6F47",
        marker_line_color="#A88454",
        marker_line_width=1,
    ))
    _apply_vintage_layout(fig, xaxis_title="Word Count", yaxis_title="Articles")
    return fig


def _scatter_topic_confidence(articles: List[Dict[str, Any]]) -> go.Figure:
    rows = [
        {
            "Title": a.get("title", "Untitled")[:40],
            "Words": a.get("word_count", 0),
            "Confidence": a.get("category", {}).get("confidence", 0),
            "Topic": a.get("category", {}).get("label", "Unknown"),
        }
        for a in articles
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return go.Figure()
    fig = px.scatter(
        df, x="Words", y="Confidence", color="Topic",
        hover_name="Title", size="Words", size_max=18,
        color_discrete_sequence=TOPIC_COLORS,
    )
    fig.update_traces(marker=dict(line=dict(color="#A88454", width=1)))
    _apply_vintage_layout(fig)
    return fig


def _bar_sentiment_by_article(articles: List[Dict[str, Any]]) -> go.Figure:
    rows = [
        {
            "Article": a.get("title", "Untitled")[:40],
            "Confidence": a.get("sentiment", {}).get("confidence", 0),
            "Sentiment": a.get("sentiment", {}).get("label", "NEUTRAL"),
        }
        for a in articles
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return go.Figure()
    fig = px.bar(
        df.sort_values("Confidence", ascending=True),
        x="Confidence", y="Article", orientation="h",
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS,
    )
    fig.update_traces(marker_line_color="#A88454", marker_line_width=0.5)
    _apply_vintage_layout(fig, yaxis={"categoryorder": "total ascending"})
    return fig


def _bar_entity_breakdown(ent_analytics: Dict[str, Any]) -> go.Figure:
    """Grouped horizontal bar of top people / orgs / locations."""
    names, counts, types = [], [], []
    for item in ent_analytics.get("most_common_people", [])[:8]:
        names.append(item["entity"]); counts.append(item["count"]); types.append("Person")
    for item in ent_analytics.get("most_common_organizations", [])[:8]:
        names.append(item["entity"]); counts.append(item["count"]); types.append("Organisation")
    for item in ent_analytics.get("most_common_locations", [])[:8]:
        names.append(item["entity"]); counts.append(item["count"]); types.append("Location")

    df = pd.DataFrame({"Entity": names, "Mentions": counts, "Type": types})
    if df.empty:
        return go.Figure()
    fig = px.bar(
        df, x="Mentions", y="Entity", color="Type", orientation="h",
        color_discrete_map={
            "Person":       "#5A3E2B",
            "Organisation": "#A36A2A",
            "Location":     "#8B6F47",
        },
    )
    fig.update_traces(marker_line_color="#A88454", marker_line_width=0.5)
    _apply_vintage_layout(fig, yaxis={"categoryorder": "total ascending"})
    return fig


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _articles_to_csv(articles: List[Dict[str, Any]]) -> bytes:
    rows = []
    for a in articles:
        cat = a.get("category", {})
        sent = a.get("sentiment", {})
        summ = a.get("summaries", {})
        rows.append({
            "Title": a.get("title", ""),
            "Section": a.get("section", ""),
            "Start Page": a.get("start_page", ""),
            "End Page": a.get("end_page", ""),
            "Word Count": a.get("word_count", 0),
            "Reading Time (sec)": a.get("reading_time", 0),
            "Topic": cat.get("label", ""),
            "Topic Confidence": cat.get("confidence", 0),
            "Sentiment": sent.get("label", ""),
            "Sentiment Confidence": sent.get("confidence", 0),
            "Short Summary": summ.get("short_summary", ""),
            "Keywords": ", ".join(k.get("keyword", "") for k in a.get("keywords", [])),
            "Entities": ", ".join(e.get("text", "") for e in a.get("entities", [])),
        })
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _full_report_json(data: Dict[str, Any]) -> bytes:
    def _safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_safe(v) for v in obj]
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        return str(obj)

    payload = _safe({
        "metadata": data.get("metadata"),
        "article_statistics": data.get("article_statistics"),
        "magazine_summary": data.get("magazine_summary"),
        "magazine_keywords": data.get("magazine_keywords"),
        "magazine_entities": data.get("magazine_entities"),
        "magazine_topics": data.get("magazine_topics"),
        "magazine_sentiment": data.get("magazine_sentiment"),
        "report": data.get("report"),
        "processed_at": data.get("processed_at"),
        "models_used": data.get("models_used"),
        "articles": [
            {
                "title": a.get("title"),
                "section": a.get("section"),
                "start_page": a.get("start_page"),
                "end_page": a.get("end_page"),
                "word_count": a.get("word_count"),
                "reading_time": a.get("reading_time"),
                "confidence": a.get("confidence"),
                "category": a.get("category"),
                "sentiment": a.get("sentiment"),
                "summaries": a.get("summaries"),
                "keywords": a.get("keywords"),
                "entities": a.get("entities"),
            }
            for a in data.get("articles", [])
        ],
    })
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# Page: Home
# ═══════════════════════════════════════════════════════════════════════════

def render_home() -> None:
    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown("""
        <div style="
            text-align: center;
            padding: 3.5rem 2rem 2rem 2rem;
            background-color: #EED7A7;
            background-image: url('data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22300%22 height=%22300%22%3E%3Cfilter id=%22n%22%3E%3CfeTurbulence type=%22fractalNoise%22 baseFrequency=%220.72%22 numOctaves=%224%22 stitchTiles=%22stitch%22/%3E%3CfeColorMatrix type=%22saturate%22 values=%220%22/%3E%3C/filter%3E%3Crect width=%22300%22 height=%22300%22 filter=%22url(%23n)%22 opacity=%220.05%22/%3E%3C/svg%3E');
            border: 1.5px solid #A88454;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 12px rgba(59,36,22,0.10);
            position: relative;
        ">
            <div style="
                font-family: 'Cinzel', Georgia, serif;
                font-size: 0.78rem;
                letter-spacing: 4px;
                text-transform: uppercase;
                color: #A36A2A;
                margin-bottom: 12px;
            ">✦ &nbsp; Est. MMXXVI &nbsp; ✦</div>
            <h1 style="
                font-family: 'Cormorant Garamond', Georgia, serif;
                font-size: 4rem;
                font-weight: 700;
                color: #3B2416;
                margin: 0 0 8px 0;
                letter-spacing: -1px;
                line-height: 1;
            ">MagazineIQ</h1>
            <div style="
                width: 180px;
                height: 2px;
                background: linear-gradient(to right, transparent, #A36A2A, transparent);
                margin: 14px auto;
            "></div>
            <p style="
                font-family: 'EB Garamond', Georgia, serif;
                font-size: 1.2rem;
                color: #5A3E2B;
                max-width: 600px;
                margin: 0 auto;
                line-height: 1.75;
                font-style: italic;
            ">
                An Intelligent Multi-Article Magazine Analysis Platform.<br>
                Upload any magazine PDF and receive a complete AI-powered<br>
                intelligence report — <strong style="font-style:normal; color:#3B2416;">entirely offline</strong>.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Technology Cards ──────────────────────────────────────────────────
    st.markdown("""
        <h2 style="font-family:'Cormorant Garamond',Georgia,serif;
                   font-size:1.9rem; color:#3B2416; text-align:center;
                   margin-bottom:4px;">Analytical Engines</h2>
        <div style="text-align:center; color:#A36A2A; letter-spacing:8px;
                    font-size:1rem; margin-bottom:1.5rem;">✦ ✦ ✦</div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    _tech_card(c1, "PDF Parsing", "PyMuPDF", "Extracts text layer from any text-based PDF")
    _tech_card(c2, "Named Entities", "spaCy NLP", "Identifies people, places, and organisations")
    _tech_card(c3, "Summarisation", "BART Base", "Generates multi-level article summaries")
    _tech_card(c4, "Classification", "DistilBERT", "Zero-shot topic classification across 12 domains")

    st.markdown("---")

    # ── Capabilities ──────────────────────────────────────────────────────
    st.markdown("""
        <h2 style="font-family:'Cormorant Garamond',Georgia,serif;
                   font-size:1.9rem; color:#3B2416; text-align:center;
                   margin-bottom:4px;">Platform Capabilities</h2>
        <div style="text-align:center; color:#A36A2A; letter-spacing:8px;
                    font-size:1rem; margin-bottom:1.5rem;">✦ ✦ ✦</div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown("""
            <div style="background:#EED7A7; border:1.5px solid #A88454; border-radius:12px;
                        padding:1.5rem 2rem; box-shadow:0 2px 6px rgba(59,36,22,0.08);">
                <p style="font-family:'EB Garamond',Georgia,serif;
                           font-size:1.05rem; color:#3B2416; line-height:2;">
                    ✦ &ensp;<strong>Article Detection</strong> — automatic boundary segmentation<br>
                    ✦ &ensp;<strong>Multi-level Summaries</strong> — short, medium &amp; detailed<br>
                    ✦ &ensp;<strong>Keyword Extraction</strong> — YAKE with magazine-level aggregation<br>
                    ✦ &ensp;<strong>Named Entity Recognition</strong> — people, organisations, locations
                </p>
            </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("""
            <div style="background:#EED7A7; border:1.5px solid #A88454; border-radius:12px;
                        padding:1.5rem 2rem; box-shadow:0 2px 6px rgba(59,36,22,0.08);">
                <p style="font-family:'EB Garamond',Georgia,serif;
                           font-size:1.05rem; color:#3B2416; line-height:2;">
                    ✦ &ensp;<strong>Zero-shot Topic Classification</strong> — 12 candidate categories<br>
                    ✦ &ensp;<strong>Chunk-based Sentiment Analysis</strong> — confidence scoring<br>
                    ✦ &ensp;<strong>Analytics Dashboard</strong> — interactive Plotly visualisations<br>
                    ✦ &ensp;<strong>Full-text Search</strong> — keyword, title &amp; entity matching
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Status Banner ─────────────────────────────────────────────────────
    if st.session_state.get("active_magazine_id"):
        from src.db_helpers import get_magazine, get_articles
        mag_id = st.session_state["active_magazine_id"]
        mag = get_magazine(mag_id)
        arts = get_articles(mag_id)
        if mag:
            total_words = sum(len(a["content"].split()) for a in arts)
            st.markdown(f"""
                <div style="background:#D8B97B; border:1.5px solid #A88454; border-radius:8px;
                            padding:1rem 1.5rem; text-align:center;
                            box-shadow:0 2px 6px rgba(59,36,22,0.08);">
                    <p style="font-family:'EB Garamond',Georgia,serif; margin:0;
                               font-size:1.1rem; color:#3B2416;">
                        <strong>{mag.get('title','Untitled')}</strong> is active —&nbsp;
                        {len(arts)} articles, {total_words:,} words.
                        Navigate to <strong>Dashboard</strong> in the sidebar to view analysis.
                    </p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background:#EED7A7; border:1px dashed #A88454; border-radius:8px;
                        padding:1rem 1.5rem; text-align:center;">
                <p style="font-family:'EB Garamond',Georgia,serif; margin:0;
                           font-size:1.05rem; color:#5A3E2B; font-style:italic;">
                    Use the <strong style="font-style:normal;">Upload Magazine</strong>
                    page in the sidebar to begin your analysis.
                </p>
            </div>
        """, unsafe_allow_html=True)


def _tech_card(col, label: str, value: str, description: str) -> None:
    """Renders a vintage statistic / technology card inside a column."""
    col.markdown(f"""
        <div style="
            background-color: #F4E6C1;
            border: 1.5px solid #A88454;
            border-radius: 12px;
            padding: 1.4rem 1rem;
            text-align: center;
            box-shadow: 0 2px 6px rgba(59,36,22,0.08);
            height: 100%;
            position: relative;
        ">
            <div style="
                position: absolute; top: 0; left: 15%; right: 15%; height: 2px;
                background: linear-gradient(to right, transparent, #A36A2A, transparent);
            "></div>
            <div style="
                font-family: 'Cinzel', Georgia, serif;
                font-size: 0.68rem;
                letter-spacing: 1.5px;
                text-transform: uppercase;
                color: #A36A2A;
                margin-bottom: 6px;
            ">{label}</div>
            <div style="
                font-family: 'Cormorant Garamond', Georgia, serif;
                font-size: 1.55rem;
                font-weight: 700;
                color: #3B2416;
                line-height: 1.2;
            ">{value}</div>
            <div style="
                font-family: 'EB Garamond', Georgia, serif;
                font-size: 0.85rem;
                color: #8B6F47;
                margin-top: 6px;
                line-height: 1.4;
                font-style: italic;
            ">{description}</div>
        </div>
    """, unsafe_allow_html=True)




# ═══════════════════════════════════════════════════════════════════════════
# Page: Upload
# ═══════════════════════════════════════════════════════════════════════════

def render_upload() -> None:
    st.title("📤 Upload Magazine")
    st.caption("Upload a text-based PDF magazine. Scanned-image PDFs are not supported.")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Select a Magazine PDF", type=["pdf"],
        help=f"Text-based magazine PDF only. Max {MAX_UPLOAD_MB} MB.",
    )

    if uploaded is None:
        st.info("Please upload a PDF to begin analysis.")
        return

    size_mb = round(uploaded.size / (1024 * 1024), 2)
    st.success(f"✅ File received: **{uploaded.name}** ({size_mb} MB)")

    if size_mb > MAX_UPLOAD_MB:
        st.error(f"File exceeds the {MAX_UPLOAD_MB} MB limit.")
        return

    models = _model_settings()
    with st.expander("🛠️ Models that will be used (On-Demand)", expanded=False):
        st.caption(f"**Summariser:** `{models['summarizer']}`")
        st.caption(f"**Classifier:** `{models['classifier']}`")
        st.caption(f"**Sentiment:** `{models['sentiment']}`")

    if st.button("🚀 Parse & Import Magazine", use_container_width=True, type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        pbar = st.progress(0, text="Reading PDF file...")
        try:
            # 1. PDF Extraction
            pbar.progress(0.2, text="📄 Extracting text from PDF...")
            extractor = PDFExtractor(tmp_path)
            extraction = extractor.extract()
            if extraction["status"] != "success":
                st.error(f"PDF extraction failed: {extraction['error_message']}")
                return

            pages = extraction["pages"]
            metadata = extraction["metadata"]
            
            page_count = len(pages)
            title = metadata.get("title") or uploaded.name
            if title.lower().endswith(".pdf"):
                title = title[:-4]

            # 2. Article Detection
            pbar.progress(0.5, text="🔍 Detecting article boundaries...")
            detector = ArticleDetector(pages)
            detection = detector.detect()
            articles = detection["articles"]

            if not articles:
                st.error("No articles could be detected in the PDF.")
                return

            # 3. Store to SQLite Database
            pbar.progress(0.8, text="💾 Saving articles to database...")
            
            from src.db_helpers import save_magazine, save_article, update_magazine_stats
            
            mag_id = save_magazine(
                filename=uploaded.name,
                title=title,
                page_count=page_count
            )
            
            total_reading_time = 0
            for art in articles:
                # Estimate reading time (basic word count / 200 words per minute * 60 seconds)
                word_count = len(art["text"].split())
                reading_time_secs = max(30, int((word_count / 200) * 60))
                total_reading_time += reading_time_secs
                
                save_article(
                    magazine_id=mag_id,
                    title=art["title"],
                    content=art["text"],
                    reading_time=reading_time_secs
                )
            
            update_magazine_stats(mag_id, total_reading_time, 0.0)
            
            st.session_state["active_magazine_id"] = mag_id
            st.session_state["_redirect_to_dashboard"] = True
            pbar.progress(1.0, text="✅ Import complete!")
            st.success(
                f"✅ **Magazine imported successfully!** {len(articles)} articles detected "
                f"({total_reading_time // 60} min total reading time).  \n"
                f"Navigate to **Dashboard** in the sidebar to view analytics."
            )

        except Exception as e:
            st.error(f"Failed to import magazine: {e}")
            logger.exception(e)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# Page: Dashboard
# ═══════════════════════════════════════════════════════════════════════════

def render_dashboard() -> None:
    st.title("📊 Magazine Analytics Dashboard")

    mag_id = st.session_state.get("active_magazine_id")
    if not mag_id:
        st.warning("No magazine imported yet. Use **Upload Magazine** first.")
        return

    from src.db_helpers import get_magazine, get_articles
    import json

    mag = get_magazine(mag_id)
    if not mag:
        st.warning("Magazine not found in database.")
        return

    articles = get_articles(mag_id)
    if not articles:
        st.warning("No articles found for this magazine.")
        return

    # Parse page_count from overall_insights JSON
    page_count = 0
    try:
        insights = json.loads(mag.get("overall_insights", "{}"))
        page_count = insights.get("page_count", 0)
    except (json.JSONDecodeError, TypeError):
        pass

    # Compute basic stats from DB data
    total_articles = len(articles)
    total_words = sum(len(a["content"].split()) for a in articles)
    total_reading_secs = sum(a["reading_time"] for a in articles)
    total_reading_mins = round(total_reading_secs / 60, 1)

    # Header
    title = mag.get("title", "Magazine")
    st.subheader(f"📖 {title}")
    st.caption(f"Filename: {mag.get('filename', 'N/A')}  •  Pages: {page_count}  •  Status: {mag.get('status', 'N/A')}")

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📄 Pages", page_count)
    k2.metric("📰 Articles", total_articles)
    k3.metric("📝 Words", f"{total_words:,}")
    k4.metric("⏱ Reading Time", f"{total_reading_mins} min")

    # ── Reading Time by Article — vintage parchment card ──────────────────
    reading_data = [{"title": a["title"][:30], "minutes": round(a["reading_time"] / 60, 1)} for a in articles]
    if reading_data:

        # ── 3 KPI mini-cards (computed from existing data) ─────────────────
        all_mins   = [row["minutes"] for row in reading_data]
        avg_mins   = round(sum(all_mins) / len(all_mins), 1) if all_mins else 0
        max_mins   = max(all_mins) if all_mins else 0
        n_articles = len(reading_data)

        kc1, kc2, kc3 = st.columns(3)
        for col, label, value in [
            (kc1, "Avg. Reading Time", f"{avg_mins} min"),
            (kc2, "Longest Read",      f"{max_mins} min"),
            (kc3, "Total Articles",    str(n_articles)),
        ]:
            col.markdown(f"""
                <div style="
                    background:#F4E6C1; border:1.5px solid #A88454;
                    border-radius:10px; padding:14px 16px;
                    box-shadow:0 2px 6px rgba(59,36,22,0.08);
                    text-align:center; margin-bottom:12px;
                ">
                    <div style="font-family:'Cinzel',Georgia,serif; font-size:10px;
                                text-transform:uppercase; letter-spacing:1.2px;
                                color:#3B2416; margin-bottom:6px;">{label}</div>
                    <div style="font-family:'Cormorant Garamond',Georgia,serif;
                                font-size:30px; font-weight:700;
                                color:#6B472B; line-height:1;">{value}</div>
                </div>
            """, unsafe_allow_html=True)

        # ── Outer dark card header ─────────────────────────────────────────
        st.markdown("""
            <div style="
                background: #2C1A0E;
                border: 2px solid #5A3E2B;
                border-radius: 14px;
                padding: 1.4rem 1.4rem 0.6rem 1.4rem;
                box-shadow: 0 4px 16px rgba(59,36,22,0.30);
                position: relative;
                margin-bottom: 0.5rem;
            ">
                <span style="position:absolute; top:10px; left:14px;
                             font-size:1.1rem; color:#A36A2A; opacity:0.55;">❦</span>
                <div style="
                    font-family:'Cormorant Garamond', Georgia, serif;
                    font-weight: 700;
                    font-size: 22px;
                    color: #F4E6C1;
                    margin-bottom: 10px;
                    padding-left: 24px;
                ">Reading time by article</div>
                <div style="border-top: 1.5px solid rgba(168,132,84,0.5);
                            margin-bottom: 12px;"></div>
            </div>
        """, unsafe_allow_html=True)

        # ── Build chart with go.Bar (avoids coloraxis/Purples override) ────
        hide_x_labels = len(reading_data) > 12
        _titles  = [row["title"]  for row in reading_data]
        _minutes = [row["minutes"] for row in reading_data]

        fig_read = go.Figure(go.Bar(
            x=_titles,
            y=_minutes,
            marker_color="#6B472B",
            marker_line_color="#8B5E2B",
            marker_line_width=0.5,
            hovertemplate="<b>%{x}</b><br>%{y} min<extra></extra>",
            hoverlabel=dict(
                bgcolor="#F6E8C8",
                bordercolor="#3B2416",
                font=dict(family="EB Garamond, Georgia, serif",
                          color="#3B2416", size=13),
            ),
        ))

        fig_read.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#EED7A7",
            bargap=0.08,
            showlegend=False,
            margin=dict(t=6, b=60, l=60, r=20),
            xaxis=dict(
                title=dict(
                    text="Article",
                    font=dict(family="EB Garamond, Georgia, serif",
                              color="#3B2416", size=13),
                    standoff=14,
                ),
                tickfont=dict(family="EB Garamond, Georgia, serif",
                              color="#3B2416", size=10),
                tickangle=-45,
                showticklabels=not hide_x_labels,
                showgrid=False,
                linecolor="#A88454",
                linewidth=1.5,
                ticks="outside",
                tickcolor="#A88454",
            ),
            yaxis=dict(
                title=dict(
                    text="Minutes",
                    font=dict(family="EB Garamond, Georgia, serif",
                              color="#3B2416", size=13),
                ),
                tickfont=dict(family="EB Garamond, Georgia, serif",
                              color="#3B2416", size=12),
                gridcolor="#D6C09A",
                gridwidth=1,
                linecolor="#A88454",
                linewidth=1.5,
                zerolinecolor="#A88454",
                zerolinewidth=1,
                rangemode="tozero",
            ),
        )
        st.plotly_chart(fig_read, use_container_width=True)

        # bottom-right ornament
        st.markdown("""
            <div style="text-align:right; color:#A36A2A; opacity:0.55;
                        font-size:1.1rem; margin-top:-10px;">❦</div>
        """, unsafe_allow_html=True)




    # ── gap between the two cards ──────────────────────────────────────────
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # Word count distribution — matches reference screenshot exactly
    word_data = [{"title": a["title"][:30], "words": len(a["content"].split())} for a in articles]
    if word_data:
        import numpy as np

        # ── Custom bins matching reference image labels ─────────────────────
        _bin_edges  = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 4000, 5000]
        _bin_labels = ["0-250","250-500","500-750","750-1000","1000-1250",
                       "1250-1500","1500-1750","1750-2000","2000-2500",
                       "2500-3000","3000-4000","4000-5000"]

        _wc_values = [row["words"] for row in word_data]
        _counts, _ = np.histogram(_wc_values, bins=_bin_edges)

        # ── Outer dark card wrapper (matches screenshot border) ─────────────
        st.markdown("""
            <div style="
                background: #2C1A0E;
                border: 2px solid #5A3E2B;
                border-radius: 14px;
                padding: 1.4rem 1.4rem 0.6rem 1.4rem;
                box-shadow: 0 4px 16px rgba(59,36,22,0.30);
                position: relative;
                margin-bottom: 0.5rem;
            ">
                <span style="position:absolute; top:10px; left:14px;
                             font-size:1.1rem; color:#A36A2A; opacity:0.55;">❦</span>
                <div style="
                    font-family:'Cormorant Garamond', Georgia, serif;
                    font-weight: 700;
                    font-size: 22px;
                    color: #F4E6C1;
                    margin-bottom: 10px;
                    padding-left: 24px;
                ">Word count distribution</div>
                <div style="border-top: 1.5px solid rgba(168,132,84,0.5);
                            margin-bottom: 12px;"></div>
            </div>
        """, unsafe_allow_html=True)

        # ── Build the Plotly bar chart ─────────────────────────────────────
        fig_words = go.Figure(go.Bar(
            x=_bin_labels,
            y=_counts.tolist(),
            marker_color="#A36A2A",
            marker_line_color="#8B5E2B",
            marker_line_width=0.5,
            hovertemplate="<b>%{x}</b><br>%{y} articles<extra></extra>",
            hoverlabel=dict(
                bgcolor="#F6E8C8",
                bordercolor="#3B2416",
                font=dict(family="EB Garamond, Georgia, serif",
                          color="#3B2416", size=13),
            ),
        ))

        fig_words.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#EED7A7",
            bargap=0,
            showlegend=False,
            margin=dict(t=6, b=50, l=60, r=20),
            xaxis=dict(
                title=dict(
                    text="Word count",
                    font=dict(family="EB Garamond, Georgia, serif",
                              color="#3B2416", size=13),
                    standoff=14,
                ),
                tickfont=dict(family="EB Garamond, Georgia, serif",
                              color="#3B2416", size=11),
                tickangle=45,
                showgrid=False,
                linecolor="#A88454",
                linewidth=1.5,
                ticks="outside",
                tickcolor="#A88454",
            ),
            yaxis=dict(
                title=dict(
                    text="Number of articles",
                    font=dict(family="EB Garamond, Georgia, serif",
                              color="#3B2416", size=13),
                ),
                tickfont=dict(family="EB Garamond, Georgia, serif",
                              color="#3B2416", size=12),
                gridcolor="#D6C09A",
                gridwidth=1,
                linecolor="#A88454",
                linewidth=1.5,
                zerolinecolor="#A88454",
                zerolinewidth=1,
                rangemode="tozero",
            ),
        )

        st.plotly_chart(fig_words, use_container_width=True)

        # bottom-right ornament after the chart
        st.markdown("""
            <div style="text-align:right; color:#A36A2A; opacity:0.55;
                        font-size:1.1rem; margin-top:-10px;">❦</div>
        """, unsafe_allow_html=True)

    st.markdown("---")





    # NLP Results (if available)
    analyzed_articles = [a for a in articles if a.get("category")]
    summarized_articles = [a for a in articles if a.get("summary")]

    if analyzed_articles:
        cl, cr = st.columns(2)
        with cl:
            st.subheader("📂 Topic Distribution")
            from collections import Counter as _Counter
            topic_counts = _Counter(a["category"] for a in analyzed_articles if a["category"])
            total_t = sum(topic_counts.values())
            if total_t > 0:
                topic_dist = {k: round(v / total_t * 100, 1) for k, v in topic_counts.items()}
                st.plotly_chart(_pie_topics(topic_dist), use_container_width=True)

        with cr:
            st.subheader("💬 Sentiment Distribution")
            pos = sum(1 for a in analyzed_articles if a.get("sentiment_score", 0) > 0.5)
            neg = sum(1 for a in analyzed_articles if a.get("sentiment_score", 0) < -0.5)
            neu = len(analyzed_articles) - pos - neg
            total_s = len(analyzed_articles)
            if total_s > 0:
                sent_data = {
                    "positive_percent": round(pos / total_s * 100, 1),
                    "neutral_percent": round(neu / total_s * 100, 1),
                    "negative_percent": round(neg / total_s * 100, 1),
                    "dominant_sentiment": "POSITIVE" if pos >= neg else ("NEGATIVE" if neg > pos else "NEUTRAL"),
                    "average_sentiment_confidence": 0.8
                }
                st.plotly_chart(_bar_sentiment(sent_data), use_container_width=True)
    else:
        st.info("💡 NLP analysis not yet performed. Go to **Article Explorer** and click analysis buttons on individual articles.")

    st.markdown("---")

    # Keyword and entity tables (if available)
    all_keywords = []
    all_entities = []
    for a in articles:
        for kw in a.get("keywords", []):
            all_keywords.append(kw["phrase"])
        for ent in a.get("entities", []):
            all_entities.append({"name": ent["name"], "label": ent["label"]})

    if all_keywords or all_entities:
        kc, ec = st.columns(2)
        with kc:
            st.subheader("🔑 Top Keywords")
            from collections import Counter as _Counter2
            kw_counts = _Counter2(all_keywords)
            if kw_counts:
                kw_df = pd.DataFrame([{"keyword": k, "frequency": c} for k, c in kw_counts.most_common(15)])
                # Build with go.Bar to avoid coloraxis/Purples override
                _kw_names  = kw_df["keyword"].tolist()
                _kw_freqs  = kw_df["frequency"].tolist()
                _max_freq  = max(_kw_freqs) if _kw_freqs else 1
                # Map frequency → brown shade: low=#C6A675, high=#3B2416
                def _brown(f, mx):
                    t = f / mx
                    r = int(198 + t * (59  - 198))
                    g = int(166 + t * (36  - 166))
                    b = int(117 + t * (22  - 117))
                    return f"rgb({r},{g},{b})"
                _kw_colors = [_brown(f, _max_freq) for f in _kw_freqs]

                fig_kw = go.Figure(go.Bar(
                    x=_kw_freqs,
                    y=_kw_names,
                    orientation="h",
                    marker_color=_kw_colors,
                    marker_line_color="#A88454",
                    marker_line_width=0.5,
                    hovertemplate="<b>%{y}</b><br>%{x} mentions<extra></extra>",
                    hoverlabel=dict(
                        bgcolor="#F6E8C8",
                        bordercolor="#3B2416",
                        font=dict(family="EB Garamond, Georgia, serif",
                                  color="#3B2416", size=12),
                    ),
                ))
                fig_kw.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#EED7A7",
                    showlegend=False,
                    margin=dict(t=10, b=20, l=20, r=20),
                    yaxis=dict(
                        categoryorder="total ascending",
                        tickfont=dict(family="EB Garamond, Georgia, serif",
                                      color="#3B2416", size=11),
                        linecolor="#A88454",
                        gridcolor="#D6C09A",
                    ),
                    xaxis=dict(
                        tickfont=dict(family="EB Garamond, Georgia, serif",
                                      color="#3B2416", size=11),
                        gridcolor="#D6C09A",
                        linecolor="#A88454",
                        zerolinecolor="#A88454",
                    ),
                )
                st.plotly_chart(fig_kw, use_container_width=True)

        with ec:
            st.subheader("🏷️ Top Entities")
            if all_entities:
                from collections import Counter as _Counter3
                ent_counts = _Counter3(e["name"] for e in all_entities)
                ent_df = pd.DataFrame([{"entity": n, "count": c} for n, c in ent_counts.most_common(15)])
                st.dataframe(ent_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Recent articles preview
    st.subheader("📰 Recent Articles")
    for a in articles[:5]:
        preview = a["content"][:200] + "..." if len(a["content"]) > 200 else a["content"]
        status_tag = ""
        if a.get("summary"):
            status_tag += " ✅ Summarized"
        if a.get("category"):
            status_tag += f" | 📂 {a['category']}"

        with st.expander(f"📄 **{a['title']}** — {len(a['content'].split()):,} words{status_tag}"):
            st.write(preview)
            st.caption(f"Reading time: {round(a['reading_time'] / 60, 1)} min")


# ═══════════════════════════════════════════════════════════════════════════
# Page: Article Explorer
# ═══════════════════════════════════════════════════════════════════════════

def render_article_explorer() -> None:
    st.title("📰 Article Explorer")

    mag_id = st.session_state.get("active_magazine_id")
    if not mag_id:
        st.warning("No magazine imported. Use **Upload Magazine** first.")
        return

    from src.db_helpers import (
        get_articles, save_article_summary, save_article_keywords,
        save_article_entities, save_article_topic_sentiment
    )

    articles = get_articles(mag_id)
    if not articles:
        st.warning("No articles found for this magazine.")
        return

    # Sort controls
    fc1, fc2 = st.columns(2)
    sort_by = fc1.selectbox("Sort by", ["Title", "Word Count ↓", "Reading Time ↓"], key="explorer_sort")
    search_q = fc2.text_input("Filter by title", key="explorer_filter")

    # Apply filtering
    if search_q.strip():
        articles = [a for a in articles if search_q.lower() in a["title"].lower()]

    # Apply sorting
    if sort_by == "Word Count ↓":
        articles.sort(key=lambda a: len(a["content"].split()), reverse=True)
    elif sort_by == "Reading Time ↓":
        articles.sort(key=lambda a: a["reading_time"], reverse=True)
    else:
        articles.sort(key=lambda a: a["title"].lower())

    st.caption(f"Showing **{len(articles)}** articles")
    st.markdown("---")

    for idx, art in enumerate(articles):
        art_id = art["id"]
        title = art["title"]
        content = art["content"]
        word_count = len(content.split())
        reading_mins = round(art["reading_time"] / 60, 1)

        # Status badges
        badges = []
        if art.get("summary"):
            badges.append("✅ Summarized")
        if art.get("category"):
            badges.append(f"📂 {art['category']}")
        if art.get("keywords"):
            badges.append(f"🔑 {len(art['keywords'])} keywords")
        if art.get("entities"):
            badges.append(f"🏷️ {len(art['entities'])} entities")
        badge_str = " | ".join(badges) if badges else "⏳ Not yet analyzed"

        with st.expander(f"**{title}** — {word_count:,} words | {reading_mins} min | {badge_str}"):
            # Preview
            preview = content[:300] + "..." if len(content) > 300 else content
            st.write(preview)
            st.markdown("---")

            # Action buttons row
            bc1, bc2, bc3, bc4, bc5 = st.columns(5)

            # --- SUMMARIZE ---
            if bc1.button("📝 Generate Summary", key=f"sum_{art_id}_{idx}"):
                if art.get("summary"):
                    st.info("Summary already generated (cached).")
                else:
                    with st.spinner("Generating summary (TextRank)..."):
                        try:
                            from src.summarizer import Summarizer
                            summarizer = Summarizer(progress_callback=None)
                            result = summarizer._build_summary_dict(content)
                            import json
                            save_article_summary(art_id, json.dumps(result))
                            st.success("✅ Summary generated and cached!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Summarization failed: {e}")

            # --- KEYWORDS & ENTITIES ---
            if bc2.button("🔑 Keywords & Entities", key=f"kw_{art_id}_{idx}"):
                if art.get("keywords") and art.get("entities"):
                    st.info("Keywords & entities already extracted (cached).")
                else:
                    with st.spinner("Extracting keywords and entities..."):
                        try:
                            from src.feature_engineering import KeywordExtractor
                            from src.ner import EntityExtractor

                            kw_ext = KeywordExtractor()
                            kws = kw_ext._extract_from_text(content)
                            kw_data = [{"phrase": k.get("keyword", ""), "score": k.get("score", 0.0)} for k in kws]
                            save_article_keywords(art_id, kw_data)

                            ent_ext = EntityExtractor()
                            ents = ent_ext._extract_from_text(content)
                            ent_data = [{"name": e.get("text", ""), "label": e.get("label", ""), "frequency": 1} for e in ents]
                            save_article_entities(art_id, ent_data)

                            st.success("✅ Keywords & entities extracted and cached!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Extraction failed: {e}")

            # --- TOPIC & SENTIMENT ---
            if bc3.button("📊 Topic & Sentiment", key=f"ts_{art_id}_{idx}"):
                if art.get("category"):
                    st.info("Topic & sentiment already analyzed (cached).")
                else:
                    with st.spinner("Classifying topic and analyzing sentiment..."):
                        try:
                            from src.topic_modeling import TopicClassifier
                            from src.sentiment import SentimentAnalyzer

                            classifier = TopicClassifier()
                            topic_result = classifier._classify_text(content)
                            category = topic_result.get("label", "Unknown")

                            analyzer = SentimentAnalyzer()
                            sent_result = analyzer._analyze_text(content)
                            sentiment_score = sent_result.get("confidence", 0.0)
                            if sent_result.get("label") == "NEGATIVE":
                                sentiment_score = -sentiment_score

                            save_article_topic_sentiment(art_id, category, sentiment_score)
                            st.success(f"✅ Topic: {category} | Sentiment: {sent_result.get('label', 'NEUTRAL')}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")

            # --- VIEW FULL TEXT ---
            if bc4.button("📄 Full Text", key=f"txt_{art_id}_{idx}"):
                st.text_area("Full Article Text", content, height=400, disabled=True, key=f"ta_{art_id}")

            # --- EXPLORE AI (XAI & Recs) ---
            if bc5.button("🤖 Explore AI", key=f"xai_{art_id}_{idx}"):
                st.session_state[f"show_xai_{art_id}"] = not st.session_state.get(f"show_xai_{art_id}", False)
                
            # Display cached results
            if art.get("summary"):
                st.markdown("---")
                st.subheader("📝 Summary")
                try:
                    import json
                    summ = json.loads(art["summary"])
                    t1, t2, t3 = st.tabs(["Short", "Medium", "Detailed"])
                    t1.write(summ.get("short_summary", "N/A"))
                    t2.write(summ.get("medium_summary", "N/A"))
                    t3.write(summ.get("detailed_summary", "N/A"))
                except (json.JSONDecodeError, TypeError):
                    st.write(art["summary"])

            if art.get("keywords"):
                st.markdown("**🔑 Keywords:**")
                kw_str = ", ".join(k["phrase"] for k in art["keywords"][:10])
                st.caption(kw_str)

            if art.get("entities"):
                st.markdown("**🏷️ Entities:**")
                ent_str = ", ".join(f"{e['name']} ({e['label']})" for e in art["entities"][:10])
                st.caption(ent_str)

            if art.get("category"):
                st.caption(f"**📂 Topic:** {art['category']}  |  **💬 Sentiment Score:** {art.get('sentiment_score', 0):.2f}")
                
            if st.session_state.get(f"show_xai_{art_id}", False):
                st.markdown("---")
                st.subheader("🤖 Explainable AI & Recommendations")
                
                from src.similarity_engine import SimilarityEngine
                engine = SimilarityEngine()
                
                # 1. XAI: Why this topic?
                cat = art.get("category")
                if cat:
                    st.markdown(f"**Why did the AI predict `{cat}`?**")
                    kws = [k["phrase"] for k in art.get("keywords", [])[:10]]
                    if kws:
                        # Rank keywords by similarity to category label
                        reasons = []
                        for k in kws:
                            sim = engine.compute_similarity(k, cat)
                            reasons.append((k, sim))
                        reasons.sort(key=lambda x: x[1], reverse=True)
                        
                        top_words = [f"`{r[0]}` ({(r[1]*100):.1f}%)" for r in reasons[:5]]
                        st.write(f"The model detected high semantic correlation between `{cat}` and these extracted keywords:")
                        st.write(", ".join(top_words))
                    else:
                        st.info("Extract keywords first to see AI explanations.")
                
                # 2. Similar Articles
                st.markdown("**Similar Articles in this Magazine:**")
                recs = engine.recommend_similar(idx, articles, top_k=3)
                if recs:
                    for r in recs:
                        sim_pct = round(r['similarity'] * 100, 1)
                        st.markdown(f"- **{r['title']}** (Similarity: {sim_pct}%)")
                        if r['reason']:
                            st.caption(f"  *Reason: {r['reason']}*")
                else:
                    st.write("Not enough articles analyzed yet to find similarities.")


# ═══════════════════════════════════════════════════════════════════════════
# Page: Search
# ═══════════════════════════════════════════════════════════════════════════

def render_search() -> None:
    st.title("🔍 Search Articles")

    mag_id = st.session_state.get("active_magazine_id")
    if not mag_id:
        st.warning("No magazine imported. Use **Upload Magazine** first.")
        return

    from src.db_helpers import get_articles
    articles = get_articles(mag_id)

    query = st.text_input("Search across all articles", placeholder="Enter a keyword or phrase...")

    if not query.strip():
        st.info("Enter a search term to find matching articles (searches text, titles, keywords, entities).")
        return

    ql = query.lower().strip()
    matches: List[Dict[str, Any]] = []

    for art in articles:
        text = art.get("content", "").lower()
        title_l = art.get("title", "").lower()
        kw_l = [k.get("phrase", "").lower() for k in art.get("keywords", [])]
        ent_l = [e.get("name", "").lower() for e in art.get("entities", [])]

        occ = text.count(ql)
        in_title = ql in title_l
        in_kw = any(ql in k for k in kw_l)
        in_ent = any(ql in e for e in ent_l)

        if occ > 0 or in_title or in_kw or in_ent:
            relevance = occ + (50 if in_title else 0) + (20 if in_kw else 0) + (10 if in_ent else 0)
            matches.append({
                "article": art, "occurrences": occ,
                "in_title": in_title, "in_keywords": in_kw, "in_entities": in_ent,
                "relevance": relevance,
            })

    matches.sort(key=lambda m: m["relevance"], reverse=True)
    st.markdown(f"**{len(matches)} article(s) found** for `{query}`")
    st.markdown("---")

    for m in matches:
        art = m["article"]
        title = art.get("title", "Untitled")

        tags = []
        if m["in_title"]:
            tags.append("🏷️ title")
        if m["in_keywords"]:
            tags.append("🔑 keywords")
        if m["in_entities"]:
            tags.append("🏷️ entities")

        tag_str = "  •  ".join(tags)
        cat_label = art.get("category", "Unknown") or "Unknown"

        with st.expander(
            f"📄 **{title}** — {m['occurrences']} occurrence(s)"
            f"{'  •  ' + tag_str if tag_str else ''}"
        ):
            text = art.get("content", "")
            idx = text.lower().find(ql)
            if idx != -1:
                start = max(0, idx - 200)
                end = min(len(text), idx + 400)
                snip = text[start:end]
                mi = idx - start
                highlighted = snip[:mi] + f"**{snip[mi:mi+len(query)]}**" + snip[mi+len(query):]
                st.markdown(f"...{highlighted}...")

            word_count = len(art.get("content", "").split())
            mc1, mc2, mc3 = st.columns(3)
            mc1.caption(f"📝 {word_count:,} words")
            mc2.caption(f"📂 {cat_label}")
            mc3.caption(f"⏱ {round(art.get('reading_time', 0) / 60, 1)} min")

            if art.get("summary"):
                try:
                    import json
                    summ = json.loads(art["summary"])
                    st.markdown(f"*{summ.get('short_summary', '')}*")
                except (json.JSONDecodeError, TypeError):
                    st.markdown(f"*{art['summary'][:200]}*")


# ═══════════════════════════════════════════════════════════════════════════
# Page: Settings
# ═══════════════════════════════════════════════════════════════════════════

def render_settings() -> None:
    st.title("⚙️ Settings")
    st.markdown("---")

    # Summariser
    st.subheader("📝 Summarisation Model")
    sum_opts = ["facebook/bart-base"]
    cur_sum = st.session_state.get("cfg_summarizer", sum_opts[0])
    sel_sum = st.selectbox(
        "Summarisation model", sum_opts,
        index=sum_opts.index(cur_sum) if cur_sum in sum_opts else 0,
        help="Model used for article and magazine summarisation.",
        key="sel_summarizer",
    )
    st.session_state["cfg_summarizer"] = sel_sum

    # Classifier
    st.subheader("📂 Topic Classification Model")
    cls_opts = ["typeform/distilbert-base-uncased-mnli"]
    cur_cls = st.session_state.get("cfg_classifier", cls_opts[0])
    sel_cls = st.selectbox(
        "Classification model", cls_opts,
        index=cls_opts.index(cur_cls) if cur_cls in cls_opts else 0,
        help="Model for zero-shot topic classification.",
        key="sel_classifier",
    )
    st.session_state["cfg_classifier"] = sel_cls

    # Sentiment
    st.subheader("💬 Sentiment Analysis Model")
    sen_opts = ["distilbert-base-uncased-finetuned-sst-2-english"]
    cur_sen = st.session_state.get("cfg_sentiment", sen_opts[0])
    sel_sen = st.selectbox(
        "Sentiment model", sen_opts,
        index=sen_opts.index(cur_sen) if cur_sen in sen_opts else 0,
        help="Model for article sentiment analysis.",
        key="sel_sentiment",
    )
    st.session_state["cfg_sentiment"] = sel_sen

    st.markdown("---")

    # Active configuration table
    st.subheader("🔧 Active Configuration")
    models = _model_settings()
    st.dataframe(
        pd.DataFrame([
            {"Module": "Summariser", "Model": models["summarizer"]},
            {"Module": "Topic Classifier", "Model": models["classifier"]},
            {"Module": "Sentiment Analyser", "Model": models["sentiment"]},
            {"Module": "Keyword Extractor", "Model": "YAKE (rule-based)"},
            {"Module": "Entity Extractor", "Model": "spaCy en_core_web_sm"},
            {"Module": "PDF Parser", "Model": "PyMuPDF (fitz)"},
        ]),
        use_container_width=True, hide_index=True,
    )
    st.info("Model changes take effect on the **next magazine upload**.")

    st.markdown("---")
    st.subheader("🗑️ Data Management")
    
    from src.db_helpers import get_all_magazines, delete_magazine
    all_mags = get_all_magazines()
    if all_mags:
        mag_delete_options = {f"{m['title']} (ID: {m['id']})": m["id"] for m in all_mags}
        selected_to_delete = st.selectbox("Select Magazine to Delete from Database", list(mag_delete_options.keys()))
        
        if st.button("🗑️ Delete Selected Magazine", type="primary"):
            try:
                to_delete_id = mag_delete_options[selected_to_delete]
                delete_magazine(to_delete_id)
                
                # If deleted was active, pop it so app.py can re-initialize to the next latest
                if st.session_state.get("active_magazine_id") == to_delete_id:
                    st.session_state.pop("active_magazine_id", None)
                    
                st.success(f"Successfully deleted '{selected_to_delete}' from database!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to delete magazine: {e}")
    else:
        st.info("No magazines currently found in the database.")


# ═══════════════════════════════════════════════════════════════════════════
# Page: About
# ═══════════════════════════════════════════════════════════════════════════

def render_about() -> None:
    st.title("ℹ️ About MagazineIQ")
    st.markdown("---")
    st.markdown("""
**MagazineIQ** is an M.Sc. Data Science final-year project.

### Project Title
*Intelligent Multi-Article Magazine Analyzer using Natural Language Processing*

### Architecture

| Layer | Technology | Module |
|---|---|---|
| Frontend | Streamlit | `dashboard.py` |
| PDF Parsing | PyMuPDF (fitz) | `PDFExtractor` |
| Article Detection | Rule-based Heuristics | `ArticleDetector` |
| Keyword Extraction | YAKE | `KeywordExtractor` |
| Named Entity Recognition | spaCy en_core_web_sm | `EntityExtractor` |
| Summarisation | BART Base (local) | `Summarizer` |
| Topic Classification | DistilBERT MNLI (zero-shot) | `TopicClassifier` |
| Sentiment Analysis | DistilBERT SST-2 (local) | `SentimentAnalyzer` |
| Analytics Engine | Pure Python | `AnalyticsEngine` |
| Visualisation | Plotly | Dashboard charts |

### NLP Pipeline Flow

```
PDF → PDFExtractor → ArticleDetector → KeywordExtractor
    → EntityExtractor → Summarizer → TopicClassifier
    → SentimentAnalyzer → AnalyticsEngine → Dashboard
```

### Privacy
All processing is performed **entirely offline**. No data leaves the machine.

### Requirements
- Python 3.9+
- 8 GB RAM minimum (16 GB recommended)
- `streamlit`, `PyMuPDF`, `spacy`, `transformers`, `yake`, `pandas`, `plotly`

### Constraints
- Scanned-image PDFs (without embedded text) are not supported
- Very large PDFs (>200 MB) may cause memory issues
- First-time model downloads require an internet connection
    """)
    st.markdown("---")
    st.caption("Built with Streamlit, HuggingFace Transformers, and spaCy.")
