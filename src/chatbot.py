import streamlit as st
import logging
from typing import List, Dict, Any, Tuple
import os

logger = logging.getLogger(__name__)

@st.cache_resource
def load_models():
    """Loads only the lightweight sentence transformer, cached in memory."""
    from sentence_transformers import SentenceTransformer
    with st.spinner("Loading lightweight semantic search model..."):
        # Very lightweight embedding model (approx 80MB)
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return embedder

@st.cache_resource
def load_generative_model():
    """Loads a small local LLM for RAG generation."""
    from transformers import pipeline
    with st.spinner("Loading generative AI model (this may take a moment on first run)..."):
        # flan-t5-base is a good balance of size (~900MB) and instruction-following ability
        generator = pipeline("text2text-generation", model="google/flan-t5-base")
    return generator

def retrieve_relevant_context(question: str, articles: List[Dict[str, Any]], embedder) -> Tuple[str, str, float]:
    """Finds the most relevant article text for a given question using semantic search."""
    from sentence_transformers import util
    import torch
    
    if not articles:
        return "", "No articles found.", 0.0
        
    # Split articles into smaller chunks for better accuracy
    chunks = []
    chunk_meta = []
    
    for art in articles:
        text = art.get("content", "")
        if not text:
            continue
            
        words = text.split()
        chunk_size = 100  # Smaller chunks for more precise highlighting
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)
            chunk_meta.append(art)
            
    if not chunks:
        return "", "Articles are empty.", 0.0

    # Embed chunks and question
    chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    
    # Calculate cosine similarities
    cos_scores = util.cos_sim(question_embedding, chunk_embeddings)[0]
    
    # Get the top match
    best_idx = torch.argmax(cos_scores).item()
    best_score = cos_scores[best_idx].item()
    
    return chunks[best_idx], chunk_meta[best_idx], best_score

def render_chatbot():
    st.header("Ask the Archive 🤖")
    st.markdown("Ask a question, and the AI will use **Retrieval-Augmented Generation (RAG)** to read the magazine and write a natural answer for you.")
    
    from src.db_helpers import get_magazine, get_articles
    
    active_id = st.session_state.get("active_magazine_id")
    if not active_id:
        st.warning("Please upload or select a magazine in the Dashboard first.")
        return
        
    magazine = get_magazine(active_id)
    st.subheader(f"📖 Searching in: {magazine.get('title')}")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # React to user input
    if prompt := st.chat_input("Ask a question about this magazine..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            try:
                # 1. Load local model (very fast)
                embedder = load_models()
                
                # 2. Get articles for active magazine
                articles = get_articles(active_id)
                
                # 3. Retrieve context
                with st.spinner("Instantly searching the archive..."):
                    context, article, score = retrieve_relevant_context(prompt, articles, embedder)
                    
                if not context:
                    st.error("I couldn't find any text to search through in this magazine.")
                else:
                    # 4. Load generative model
                    generator = load_generative_model()
                    
                    # 5. Construct prompt and generate answer
                    with st.spinner("Thinking and writing an answer..."):
                        rag_prompt = f"Answer the following question based on the provided context.\n\nContext: {context}\n\nQuestion: {prompt}\n\nAnswer:"
                        gen_output = generator(rag_prompt, max_length=150, num_beams=2, early_stopping=True)
                        generated_answer = gen_output[0]['generated_text']

                    source_title = article.get("title", "Unknown Article")
                    reading_time = article.get("reading_time", 0)
                    category = article.get("category") or "Uncategorized"
                    
                    response = f"**AI Answer:**\n{generated_answer}\n\n"
                    response += f"---\n*Based on information from the article **'{source_title}'** (Confidence: {score:.2f}).*\n\n"
                    
                    response += f"**Article Details:**\n"
                    response += f"- **Topic:** {category}\n"
                    if reading_time > 0:
                        response += f"- **Reading Time:** {max(1, reading_time // 60)} mins\n"
                        
                    summary = article.get("summary")
                    if summary:
                        if isinstance(summary, str) and summary.startswith('{'):
                            import json
                            try:
                                sum_dict = json.loads(summary)
                                short_sum = sum_dict.get("short_summary")
                                if short_sum:
                                    response += f"- **Summary:** {short_sum}\n"
                            except:
                                response += f"- **Summary:** {summary}\n"
                        else:
                            response += f"- **Summary:** {summary}\n"
                    else:
                        fallback_text = article.get("content", "")[:250].strip()
                        if fallback_text:
                            response += f"- **Summary:** *(Not yet generated)* Preview: {fallback_text}...\n"
                            
                    response += f"\n**Relevant Excerpt:**\n> \"...{context}...\"\n\n"
                    response += "Does this help answer your question?"
                    
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
            except Exception as e:
                logger.error(f"Chatbot error: {e}")
                st.error(f"An error occurred while generating the answer: {e}")
