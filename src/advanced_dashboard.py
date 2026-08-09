import streamlit as st
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Caching Functions (Speeds up loading significantly)
# ---------------------------------------------------------------------------

@st.cache_data
def get_cached_intelligence_score(magazine_id: int, _articles: list):
    from src.scoring_engine import ScoringEngine
    engine = ScoringEngine()
    return engine.compute_intelligence_score(_articles)

@st.cache_data
def get_cached_magazine_dna(magazine_id: int, _articles: list):
    from src.scoring_engine import ScoringEngine
    engine = ScoringEngine()
    return engine.compute_magazine_dna(_articles)

@st.cache_data
def get_cached_knowledge_graph(magazine_id: int, _articles: list) -> str:
    from src.visualization_engine import generate_network_html
    nodes_dict = {}
    edges_dict = {}
    
    for art in _articles:
        ents = [e for e in art.get("entities", []) if e.get("label") in ["PERSON", "ORG", "GPE", "LOC"]]
        ents = sorted(ents, key=lambda x: x.get("frequency", 1), reverse=True)[:10]
        
        for e in ents:
            name = e["name"]
            group = "Person" if e["label"] == "PERSON" else "Organisation" if e["label"] == "ORG" else "Location"
            if name not in nodes_dict:
                nodes_dict[name] = {"id": name, "group": group, "size": 10}
            else:
                nodes_dict[name]["size"] = min(30, nodes_dict[name]["size"] + 2)
                
        for i in range(len(ents)):
            for j in range(i+1, len(ents)):
                e1 = ents[i]["name"]
                e2 = ents[j]["name"]
                if e1 == e2: continue
                edge_tuple = tuple(sorted([e1, e2]))
                if edge_tuple not in edges_dict:
                    edges_dict[edge_tuple] = 1
                else:
                    edges_dict[edge_tuple] += 1
                    
    nodes = list(nodes_dict.values())
    edges = [(e[0], e[1], w) for e, w in edges_dict.items()]
    
    if len(nodes) > 100:
        edges = [e for e in edges if e[2] > 1]
        connected_nodes = set([e[0] for e in edges] + [e[1] for e in edges])
        nodes = [n for n in nodes if n["id"] in connected_nodes]
        
    return generate_network_html(nodes, edges)

@st.cache_data
def get_cached_keyword_network(magazine_id: int, _articles: list) -> str:
    from src.visualization_engine import generate_network_html
    nodes_dict = {}
    edges_dict = {}
    
    for art in _articles:
        kws = [k.get("phrase", "") for k in art.get("keywords", [])[:7]]
        for k in kws:
            if not k: continue
            if k not in nodes_dict:
                nodes_dict[k] = {"id": k, "group": "Keyword", "size": 15}
            else:
                nodes_dict[k]["size"] = min(40, nodes_dict[k]["size"] + 3)
                
        for i in range(len(kws)):
            for j in range(i+1, len(kws)):
                k1 = kws[i]
                k2 = kws[j]
                if not k1 or not k2 or k1 == k2: continue
                edge = tuple(sorted([k1, k2]))
                if edge not in edges_dict:
                    edges_dict[edge] = 1
                else:
                    edges_dict[edge] += 1
                    
    nodes = list(nodes_dict.values())
    edges = [(e[0], e[1], w) for e, w in edges_dict.items() if w > 1]
    
    return generate_network_html(nodes, edges)

@st.cache_resource
def get_cached_similarity_engine():
    from src.similarity_engine import SimilarityEngine
    return SimilarityEngine()

@st.cache_resource
def get_indexed_similarity_engine(magazine_id: int, _articles: list):
    engine = get_cached_similarity_engine()
    engine.build_article_index(_articles)
    return engine


# ---------------------------------------------------------------------------
# View helper
# ---------------------------------------------------------------------------

def _get_active_data():
    from src.db_helpers import get_magazine, get_articles
    mag_id = st.session_state.get("active_magazine_id")
    if not mag_id:
        st.warning("Please upload or select a magazine first from the Home page.")
        return None, None
    mag = get_magazine(mag_id)
    articles = get_articles(mag_id)
    if not mag or not articles:
        st.error("Could not load magazine data. Please re-analyze the PDF.")
        return None, None
    return mag, articles


# ---------------------------------------------------------------------------
# Magazine Intelligence
# ---------------------------------------------------------------------------
def render_intelligence():
    from src.visualization_engine import plot_sunburst_intelligence

    st.title("🧠 Magazine Intelligence Score")
    st.caption("Advanced AI metrics and offline analysis of the active magazine.")
    
    mag, articles = _get_active_data()
    if not mag: return
    
    with st.spinner("Computing Intelligence Score..."):
        score_data = get_cached_intelligence_score(mag["id"], articles)
        
    overall = score_data.get("overall_score", 0)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div style="background-color: #EED7A7; border: 2px solid #A88454; border-radius: 10px; padding: 2rem; text-align: center; margin-top: 2rem;">
            <h3 style="color: #5A3E2B; margin-bottom: 0;">Overall Score</h3>
            <h1 style="color: #3B2416; font-size: 4rem; margin: 0;">{overall}</h1>
            <p style="color: #8B6F47;">/ 100</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.plotly_chart(plot_sunburst_intelligence(score_data), use_container_width=True)

# ---------------------------------------------------------------------------
# Magazine DNA
# ---------------------------------------------------------------------------
def render_dna():
    from src.visualization_engine import plot_radar_dna

    st.title("🧬 Magazine DNA")
    st.caption("Topic breakdown mapped to 8 core research categories.")
    
    mag, articles = _get_active_data()
    if not mag: return
    
    with st.spinner("Computing Magazine DNA..."):
        dna_scores = get_cached_magazine_dna(mag["id"], articles)
        
    st.plotly_chart(plot_radar_dna(dna_scores), use_container_width=True)

# ---------------------------------------------------------------------------
# Knowledge Graph
# ---------------------------------------------------------------------------
def render_knowledge_graph():
    import streamlit.components.v1 as components

    st.title("🌐 Knowledge Graph")
    st.caption("Interactive co-occurrence network of People, Organizations, and Locations.")
    
    mag, articles = _get_active_data()
    if not mag: return
    
    with st.spinner("Building entity relationship graph..."):
        html_content = get_cached_knowledge_graph(mag["id"], articles)
        components.html(html_content, height=550)

# ---------------------------------------------------------------------------
# Semantic Search
# ---------------------------------------------------------------------------
def render_semantic_search():
    st.title("🔍 Semantic Search")
    st.caption("Concept-based retrieval using FAISS and local embeddings.")
    
    mag, articles = _get_active_data()
    if not mag: return
    
    query = st.text_input("Enter a concept or phrase (e.g. 'Artificial Intelligence')")
    
    if query:
        with st.spinner("Building/Loading FAISS Index..."):
            engine = get_indexed_similarity_engine(mag["id"], articles)
                
        with st.spinner("Searching..."):
            results = engine.semantic_search(query, top_k=5)
            
        if not results:
            st.info("No matches found.")
        else:
            for r in results:
                score = round(r["similarity"] * 100, 1)
                st.markdown(f"#### {r['article_title']} (Similarity: {score}%)")
                st.markdown(f"> {r['text'][:500]}...")
                st.divider()

# ---------------------------------------------------------------------------
# Article Similarity & Duplicates
# ---------------------------------------------------------------------------
def render_article_similarity():
    st.title("👯 Article Similarity & Duplicates")
    st.caption("Detects highly similar and nearly identical articles.")
    
    mag, articles = _get_active_data()
    if not mag: return
    
    with st.spinner("Analyzing semantic overlap..."):
        engine = get_indexed_similarity_engine(mag["id"], articles)
        duplicates = engine.find_duplicates(articles, threshold=0.90)
        
    if not duplicates:
        st.success("✅ No duplicate articles detected (Similarity > 90%).")
    else:
        st.error(f"⚠️ Found {len(duplicates)} potentially duplicate article pairs!")
        
        for d in duplicates:
            sim = round(d["similarity"] * 100, 1)
            st.markdown(f"""
            <div style="background-color: #fce8e8; border-left: 4px solid #cc0000; padding: 1rem; margin-bottom: 1rem;">
                <h4 style="color: #cc0000; margin-top: 0;">Similarity: {sim}%</h4>
                <strong>Article A:</strong> {d['article_1_title']}<br>
                <strong>Article B:</strong> {d['article_2_title']}
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Topic Timeline
# ---------------------------------------------------------------------------
def render_topic_timeline():
    from src.visualization_engine import plot_timeline_topics

    st.title("📈 Topic Evolution Timeline")
    st.caption("Visualize how the magazine's dominant topics shift sequentially.")
    
    mag, articles = _get_active_data()
    if not mag: return
    
    st.plotly_chart(plot_timeline_topics(articles), use_container_width=True)

# ---------------------------------------------------------------------------
# Keyword Network
# ---------------------------------------------------------------------------
def render_keyword_network():
    import streamlit.components.v1 as components

    st.title("🕸️ Keyword Network")
    st.caption("Co-occurrence graph of key concepts extracted across the magazine.")
    
    mag, articles = _get_active_data()
    if not mag: return
    
    with st.spinner("Building keyword relationship graph..."):
        html_content = get_cached_keyword_network(mag["id"], articles)
        components.html(html_content, height=550)
