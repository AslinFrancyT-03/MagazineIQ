import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Any, Dict, List, Tuple
import os
import tempfile

logger = logging.getLogger(__name__)

# Vintage color scheme matches dashboard.py
_COLORS = ["#5A3E2B", "#8B6F47", "#A36A2A", "#6B5D45", "#A88454", "#C6A675", "#D8B97B", "#3B2416"]

def _apply_vintage_layout(fig: go.Figure, **overrides) -> go.Figure:
    layout = dict(
        paper_bgcolor="#F4E6C1",
        plot_bgcolor="#F4E6C1",
        font=dict(family="EB Garamond, Georgia, serif", color="#3B2416", size=12),
        title_font=dict(family="Cormorant Garamond, Georgia, serif", color="#3B2416", size=18),
        margin=dict(t=36, b=28, l=28, r=28),
    )
    layout.update(overrides)
    fig.update_layout(**layout)
    return fig

def plot_radar_dna(dna_scores: Dict[str, float]) -> go.Figure:
    """Generates a Radar Chart for Magazine DNA."""
    categories = list(dna_scores.keys())
    values = list(dna_scores.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]], # Close the loop
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(163, 106, 42, 0.4)',
        line=dict(color='#8B6F47', width=2),
        name='Magazine DNA'
    ))
    
    _apply_vintage_layout(fig, polar=dict(
        bgcolor="#EED7A7",
        radialaxis=dict(visible=True, range=[0, max(100, max(values) + 10)], gridcolor="rgba(168,132,84,0.3)"),
        angularaxis=dict(gridcolor="rgba(168,132,84,0.3)")
    ), showlegend=False)
    return fig

def plot_sunburst_intelligence(score_dict: Dict[str, Any]) -> go.Figure:
    """Generates a Sunburst Chart for Intelligence Score components."""
    components = score_dict.get("components", {})
    if not components:
        return go.Figure()
        
    labels = ["Intelligence Score"] + list(components.keys())
    parents = [""] + ["Intelligence Score"] * len(components)
    values = [sum(components.values())] + list(components.values())
    
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(colors=_COLORS * 2, line=dict(color='#F4E6C1', width=2))
    ))
    
    _apply_vintage_layout(fig, margin=dict(t=20, l=20, r=20, b=20))
    return fig

def plot_timeline_topics(articles: List[Dict[str, Any]]) -> go.Figure:
    """Generates a Timeline (Scatter) of topic evolution across pages."""
    if not articles:
        return go.Figure()
        
    rows = []
    for i, a in enumerate(articles):
        rows.append({
            "Article Index": i + 1,
            "Title": a.get("title", f"Article {i+1}")[:30] + "...",
            "Topic": a.get("category", {}).get("label", "Unknown"),
            "Word Count": a.get("word_count", 100)
        })
        
    df = pd.DataFrame(rows)
    fig = px.scatter(
        df, x="Article Index", y="Topic", 
        color="Topic", size="Word Count",
        hover_name="Title",
        color_discrete_sequence=_COLORS
    )
    
    fig.update_traces(mode='lines+markers', line=dict(color='rgba(139, 111, 71, 0.4)', width=1))
    
    _apply_vintage_layout(fig, 
        xaxis=dict(gridcolor="rgba(168,132,84,0.2)", linecolor="#A88454", title="Article Sequence"),
        yaxis=dict(gridcolor="rgba(168,132,84,0.2)", linecolor="#A88454", title="Dominant Topic")
    )
    return fig

def plot_heatmap_comparison(topics_1: Dict[str, int], topics_2: Dict[str, int]) -> go.Figure:
    """Generates a Heatmap comparing topic distributions of two magazines."""
    all_topics = sorted(list(set(topics_1.keys()).union(set(topics_2.keys()))))
    
    val1 = [topics_1.get(t, 0) for t in all_topics]
    val2 = [topics_2.get(t, 0) for t in all_topics]
    
    fig = go.Figure(data=go.Heatmap(
        z=[val1, val2],
        x=all_topics,
        y=['Magazine A', 'Magazine B'],
        colorscale=[[0.0, "#F4E6C1"], [0.5, "#A88454"], [1.0, "#3B2416"]],
        text=[[str(v) for v in val1], [str(v) for v in val2]],
        texttemplate="%{text}",
        textfont={"color": "white"}
    ))
    
    _apply_vintage_layout(fig)
    return fig

def generate_network_html(nodes: List[Dict[str, Any]], edges: List[Tuple[str, str, int]]) -> str:
    """
    Builds a NetworkX graph and renders it using PyVis.
    Returns the HTML content as a string.
    Nodes should be [{"id": str, "group": str, "size": int}]
    """
    import networkx as nx
    from pyvis.network import Network
    
    G = nx.Graph()
    for n in nodes:
        # Map group to vintage color
        color = "#8B6F47"
        if n.get("group") == "Person": color = "#5A3E2B"
        elif n.get("group") == "Organisation": color = "#A36A2A"
        elif n.get("group") == "Location": color = "#C6A675"
        
        G.add_node(n["id"], title=n["id"], label=n["id"], group=n.get("group", ""), color=color, size=n.get("size", 10))
        
    for source, target, weight in edges:
        if source in G.nodes and target in G.nodes:
            G.add_edge(source, target, value=weight, color="#A88454")
            
    # Use PyVis to generate HTML
    net = Network(height='500px', width='100%', bgcolor='#EED7A7', font_color='#3B2416')
    net.from_nx(G)
    
    # Physics options for smooth, elegant movement
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    """)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        with open(tmp.name, "r", encoding="utf-8") as f:
            html = f.read()
    
    try:
        os.unlink(tmp.name)
    except:
        pass
        
    return html
