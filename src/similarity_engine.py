import logging
from typing import Any, Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class SimilarityEngine:
    """
    Handles vector embeddings, FAISS indexing, semantic search, 
    article similarity recommendations, and duplicate detection.
    """
    
    _instance = None
    
    def __new__(cls, model_name: str = "all-MiniLM-L6-v2"):
        if cls._instance is None:
            cls._instance = super(SimilarityEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if self._initialized:
            return
            
        import faiss
        from sentence_transformers import SentenceTransformer
        
        self.model_name = model_name
        logger.info(f"Loading embedding model '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # We use inner product for cosine similarity (since embeddings will be normalized)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks_map = {} # Maps FAISS index to chunk metadata
        self._initialized = True
        
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalizes embeddings for cosine similarity via inner product."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1
        return embeddings / norms
        
    def build_article_index(self, articles: List[Dict[str, Any]]):
        """
        Builds a FAISS index from article contents for semantic search.
        Breaks articles into chunks.
        """
        import faiss
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks_map = {}
        
        if not articles:
            return
            
        all_chunks = []
        chunk_metadata = []
        
        # Simple chunking by paragraph or fixed length
        for art_idx, art in enumerate(articles):
            text = art.get("text", "")
            # Split by paragraph
            paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 50]
            if not paragraphs:
                paragraphs = [text] # fallback
                
            for p_idx, p in enumerate(paragraphs):
                all_chunks.append(p)
                chunk_metadata.append({
                    "article_index": art_idx,
                    "article_title": art.get("title", "Untitled"),
                    "text": p
                })
                
        if all_chunks:
            logger.info(f"Encoding {len(all_chunks)} chunks for FAISS...")
            embeddings = self.model.encode(all_chunks, convert_to_numpy=True)
            embeddings = self._normalize(embeddings)
            
            self.index.add(embeddings)
            self.chunks_map = {i: meta for i, meta in enumerate(chunk_metadata)}
            
    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Searches the FAISS index for the most conceptually similar chunks."""
        if self.index.ntotal == 0:
            return []
            
        q_emb = self.model.encode([query], convert_to_numpy=True)
        q_emb = self._normalize(q_emb)
        
        distances, indices = self.index.search(q_emb, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx in self.chunks_map:
                meta = self.chunks_map[idx].copy()
                meta["similarity"] = float(dist)
                results.append(meta)
                
        return results

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Computes cosine similarity between two strings."""
        if not text1 or not text2:
            return 0.0
            
        embs = self.model.encode([text1, text2], convert_to_numpy=True)
        embs = self._normalize(embs)
        
        # Inner product of normalized vectors = cosine similarity
        sim = np.dot(embs[0], embs[1])
        return float(sim)
        
    def find_duplicates(self, articles: List[Dict[str, Any]], threshold: float = 0.90) -> List[Dict[str, Any]]:
        """
        Detects nearly identical articles by comparing their embeddings.
        Returns a list of duplicate pairs with matched sections.
        """
        if len(articles) < 2:
            return []
            
        texts = [a.get("text", "") for a in articles]
        embs = self.model.encode(texts, convert_to_numpy=True)
        embs = self._normalize(embs)
        
        sim_matrix = np.dot(embs, embs.T)
        
        duplicates = []
        n = len(articles)
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])
                if sim >= threshold:
                    duplicates.append({
                        "article_1_idx": i,
                        "article_1_title": articles[i].get("title", f"Article {i+1}"),
                        "article_2_idx": j,
                        "article_2_title": articles[j].get("title", f"Article {j+1}"),
                        "similarity": sim
                    })
                    
        return duplicates
        
    def recommend_similar(self, target_idx: int, articles: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Recommends mathematically similar articles based on vector overlap.
        """
        if not articles or target_idx < 0 or target_idx >= len(articles):
            return []
            
        texts = [a.get("text", "") for a in articles]
        embs = self.model.encode(texts, convert_to_numpy=True)
        embs = self._normalize(embs)
        
        target_emb = embs[target_idx]
        similarities = np.dot(embs, target_emb)
        
        recommendations = []
        for i, sim in enumerate(similarities):
            if i == target_idx:
                continue
            
            # Find common keywords and topics for XAI explanation
            target_kws = {k.get("phrase", "").lower() for k in articles[target_idx].get("keywords", [])}
            other_kws = {k.get("phrase", "").lower() for k in articles[i].get("keywords", [])}
            common_kws = list(target_kws.intersection(other_kws))
            
            target_topic = articles[target_idx].get("category")
            other_topic = articles[i].get("category")
            
            reason = "High semantic density match."
            if target_topic == other_topic and target_topic:
                reason = f"Both articles focus extensively on {target_topic}."
            elif common_kws:
                reason = "Strong conceptual overlap based on shared key terms."
                
            recommendations.append({
                "article_idx": i,
                "title": articles[i].get("title", f"Article {i+1}"),
                "similarity": float(sim),
                "common_keywords": common_kws[:5],
                "common_topic": target_topic if target_topic == other_topic else None,
                "reason": reason
            })
            
        # Sort by highest similarity
        recommendations.sort(key=lambda x: x["similarity"], reverse=True)
        return recommendations[:top_k]
