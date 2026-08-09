import streamlit as st
import logging

logger = logging.getLogger(__name__)


def render_comparison():
    from src.db_helpers import get_all_magazines, get_articles, get_magazine
    from src.similarity_engine import SimilarityEngine
    from src.visualization_engine import plot_heatmap_comparison, plot_radar_dna
    from src.scoring_engine import ScoringEngine

    st.title("⚖️ Magazine Comparison")
    st.caption("Side-by-side analysis of two separate magazine issues.")
    
    mags = get_all_magazines()
    if len(mags) < 2:
        st.warning("You need to upload at least two magazines to compare them.")
        return
        
    mag_options = {f"{m['title']} (ID: {m['id']})": m['id'] for m in mags}
    
    col1, col2 = st.columns(2)
    with col1:
        mag_a_label = st.selectbox("Select Magazine A", list(mag_options.keys()), index=0)
    with col2:
        mag_b_label = st.selectbox("Select Magazine B", list(mag_options.keys()), index=1)
        
    mag_a_id = mag_options[mag_a_label]
    mag_b_id = mag_options[mag_b_label]
    
    if mag_a_id == mag_b_id:
        st.warning("Please select two different magazines.")
        return
        
    articles_a = get_articles(mag_a_id)
    articles_b = get_articles(mag_b_id)
    mag_a = get_magazine(mag_a_id)
    mag_b = get_magazine(mag_b_id)
    
    if not articles_a or not articles_b:
        st.error("Missing article data for one or both magazines.")
        return
        
    st.markdown("---")
    
    # AI Similarity Percentage
    engine = SimilarityEngine()
    summary_a = mag_a.get("overall_summary", " ".join([a.get("summary", "") for a in articles_a]))
    summary_b = mag_b.get("overall_summary", " ".join([a.get("summary", "") for a in articles_b]))
    
    sim_score = engine.compute_similarity(summary_a[:5000], summary_b[:5000]) # use up to 5k chars
    sim_percent = round(sim_score * 100, 1)
    
    st.markdown(f"""
    <div style="background-color: #EED7A7; border: 2px solid #A88454; border-radius: 10px; padding: 1rem; text-align: center; margin-bottom: 2rem;">
        <h4 style="color: #5A3E2B; margin-bottom: 0;">AI Similarity Percentage</h4>
        <h2 style="color: #3B2416; margin: 0;">{sim_percent}%</h2>
        <p style="color: #8B6F47; font-size: 0.9em;">(Based on Cosine Similarity of Meta-Summaries)</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    
    scoring = ScoringEngine()
    dna_a = scoring.compute_magazine_dna(articles_a)
    dna_b = scoring.compute_magazine_dna(articles_b)
    
    with c1:
        st.markdown(f"### {mag_a.get('title')}")
        st.metric("Total Articles", len(articles_a))
        score_a = scoring.compute_intelligence_score(articles_a)
        st.metric("Intelligence Score", score_a["overall_score"])
        st.plotly_chart(plot_radar_dna(dna_a), use_container_width=True)
        
    with c2:
        st.markdown(f"### {mag_b.get('title')}")
        st.metric("Total Articles", len(articles_b))
        score_b = scoring.compute_intelligence_score(articles_b)
        st.metric("Intelligence Score", score_b["overall_score"])
        st.plotly_chart(plot_radar_dna(dna_b), use_container_width=True)
        
    st.markdown("---")
    st.subheader("Topic Overlap Heatmap")
    
    def get_topic_dist(articles):
        dist = {}
        for a in articles:
            label = a.get("category")
            if label:
                dist[label] = dist.get(label, 0) + 1
        return dist
        
    st.plotly_chart(plot_heatmap_comparison(get_topic_dist(articles_a), get_topic_dist(articles_b)), use_container_width=True)
