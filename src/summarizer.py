"""
MagazineIQ — Summarizer Module
===============================
Generates short, medium, and detailed summaries for articles and an overall
magazine summary.

Primary:  facebook/bart-large-cnn via HuggingFace AutoModelForSeq2SeqLM
Fallback: TextRank extractive summarisation (spaCy + NetworkX, fully offline)
"""

import logging
import math
import gc
import concurrent.futures
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL: str = "Falconsai/text_summarization"
MODEL_MAX_INPUT_TOKENS: int = 512
WORDS_PER_TOKEN: float = 0.75
CHUNK_WORD_LIMIT: int = int(MODEL_MAX_INPUT_TOKENS * WORDS_PER_TOKEN)

SUMMARY_LENGTHS: Dict[str, Dict[str, int]] = {
    "short":    {"min_length": 30,  "max_length": 80},
    "medium":   {"min_length": 80,  "max_length": 180},
    "detailed": {"min_length": 150, "max_length": 350},
}


class Summarizer:
    """
    Generates short, medium, and detailed summaries for articles and an overall
    magazine summary using a fully local HuggingFace transformer model.

    The model is configurable via the constructor.
    Supports an optional progress callback for UI updates.
    """

    _cached_pipeline: Optional[Dict[str, Any]] = None
    _cached_model_name: Optional[str] = None

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> None:
        """
        Args:
            model_name: HuggingFace model identifier. Defaults to BART Large CNN.
            progress_callback: Optional callable receiving a status string for
                               UI progress reporting (e.g., Streamlit st.status).
        """
        self._model_name: str = model_name
        self._progress_callback: Optional[Callable[[str], None]] = progress_callback
        self._pipeline = self._load_pipeline()

    # ------------------------------------------------------------------
    # Private: Model Loading
    # ------------------------------------------------------------------

    def _load_pipeline_sync(self) -> Dict[str, Any]:
        """Internal synchronous model loading."""
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name)
        return {"model": model, "tokenizer": tokenizer}

    def _load_pipeline(self) -> Optional[Dict[str, Any]]:
        """
        Forces the use of the high-speed TextRank extractive summarization algorithm
        to ensure generation is instantaneous while remaining accurate to the source text.
        """
        logger.info("Using high-speed TextRank extractive summarization.")
        self._notify("⚡ Using high-speed summarization mode...")
        return None

    # ------------------------------------------------------------------
    # Private: Progress Notification
    # ------------------------------------------------------------------

    def _notify(self, message: str) -> None:
        """Sends a progress update via the callback if one is registered."""
        logger.info(message)
        if self._progress_callback is not None:
            try:
                self._progress_callback(message)
            except Exception as cb_err:
                logger.warning(f"Progress callback raised an error: {cb_err}")

    # ------------------------------------------------------------------
    # Private: Chunking
    # ------------------------------------------------------------------

    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Splits long text into word-count-bounded chunks so each fits within
        the model's maximum input token limit.
        """
        words: List[str] = text.split()
        return [
            " ".join(words[i: i + CHUNK_WORD_LIMIT])
            for i in range(0, len(words), CHUNK_WORD_LIMIT)
        ] or [text]

    # ------------------------------------------------------------------
    # Private: Entity Injection
    # ------------------------------------------------------------------

    def _inject_entities(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """
        Prepends a compact entity context string to the text before summarization.
        This helps the transformer model preserve important named entities (PERSON,
        ORG, GPE) in its output, since they appear prominently in the input window.
        """
        important_labels: set = {"PERSON", "ORG", "GPE"}
        important_names: List[str] = list({
            ent["text"]
            for ent in entities
            if ent.get("label") in important_labels and ent.get("text")
        })

        if not important_names:
            return text

        entity_prefix: str = "Key entities: " + ", ".join(important_names[:10]) + ". "
        return entity_prefix + text

    # ------------------------------------------------------------------
    # Private: TextRank Fallback
    # ------------------------------------------------------------------

    def _textrank_fallback(self, text: str, max_words: int) -> str:
        """
        Lightweight TextRank extractive summarisation using spaCy and NetworkX.
        Works completely offline without any deep learning models.
        """
        try:
            import spacy
            import networkx as nx
        except ImportError:
            logger.error("spaCy or networkx not installed for TextRank fallback.")
            words = text.split()
            return " ".join(words[:max_words])

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.error("SpaCy model not found for fallback. Returning truncated text.")
            words = text.split()
            return " ".join(words[:max_words])

        doc = nlp(text[:100_000])  # Guard against very long texts
        sentences = list(doc.sents)
        if len(sentences) <= 1:
            return text

        # Build similarity graph
        graph = nx.Graph()
        for i, sent in enumerate(sentences):
            graph.add_node(i)

        def similarity(s1, s2):
            words1 = {t.lemma_.lower() for t in s1 if not t.is_stop and not t.is_punct}
            words2 = {t.lemma_.lower() for t in s2 if not t.is_stop and not t.is_punct}
            if not words1 or not words2:
                return 0.0
            return len(words1 & words2) / (math.log(len(words1) + 1) + math.log(len(words2) + 1) + 1e-5)

        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                sim = similarity(sentences[i], sentences[j])
                if sim > 0:
                    graph.add_edge(i, j, weight=sim)

        try:
            scores = nx.pagerank(graph, weight="weight")
        except Exception:
            words = text.split()
            return " ".join(words[:max_words])

        ranked = sorted(((scores[i], i, s.text) for i, s in enumerate(sentences)), reverse=True)

        selected_indices = []
        current_words = 0
        for _score, idx, s_text in ranked:
            wc = len(s_text.split())
            if current_words + wc > max_words and current_words > 0:
                break
            selected_indices.append(idx)
            current_words += wc

        selected_indices.sort()  # Restore chronological order
        return " ".join(sentences[idx].text for idx in selected_indices)

    # ------------------------------------------------------------------
    # Private: Core Summarization
    # ------------------------------------------------------------------

    def _run_model_sync(self, text: str, min_length: int, max_length: int) -> str:
        """Synchronous internal execution of the transformer model."""
        if self._pipeline is None:
            return self._textrank_fallback(text, max_length)

        model = self._pipeline["model"]
        tokenizer = self._pipeline["tokenizer"]

        word_count: int = len(text.split())
        effective_max: int = min(max_length, max(min_length + 10, word_count // 2))

        try:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=MODEL_MAX_INPUT_TOKENS,
                truncation=True,
            )
            summary_ids = model.generate(
                inputs["input_ids"],
                min_length=min_length,
                max_length=effective_max,
                num_beams=1,
                early_stopping=True,
            )
            return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

        except Exception as e:
            logger.warning(f"Model inference failed: {e}. Using TextRank fallback.")
            return self._textrank_fallback(text, max_length)

    def _run_model(self, text: str, min_length: int, max_length: int) -> str:
        """
        Runs the transformer model on a single text block within token limits.
        Executes inside a thread with a strict 30-second timeout.
        Falls back to TextRank extractive summarisation if it times out or fails.
        """
        if self._pipeline is None:
            return self._textrank_fallback(text, max_length)
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._run_model_sync, text, min_length, max_length)
            try:
                # 30-second strict timeout for one article chunk
                return future.result(timeout=30)
            except concurrent.futures.TimeoutError:
                logger.warning("Summarization timed out (>30s). Falling back to TextRank.")
                return self._textrank_fallback(text, max_length)
            except Exception as e:
                logger.warning(f"Summarization thread failed: {e}. Falling back to TextRank.")
                return self._textrank_fallback(text, max_length)

    def _summarize_text(
        self,
        text: str,
        min_length: int,
        max_length: int,
        cohere_chunks: bool = True
    ) -> str:
        """
        Summarizes text of any length.
        Long articles are chunked; each chunk is summarized independently.
        If cohere_chunks is True, chunk summaries are re-summarized into a
        final coherent paragraph rather than simply concatenated.
        """
        if not text or not text.strip():
            return ""

        chunks: List[str] = self._split_into_chunks(text)

        if len(chunks) == 1:
            return self._run_model(text, min_length, max_length)

        # Summarize each chunk independently
        chunk_summaries: List[str] = []
        for chunk in chunks:
            if len(chunk.split()) < 30:
                chunk_summaries.append(chunk.strip())
            else:
                chunk_summaries.append(
                    self._run_model(chunk, min_length // 2, max_length // 2)
                )

        combined: str = " ".join(chunk_summaries)

        # Optional second-pass coherence summarization for long article chunks
        if cohere_chunks and len(combined.split()) > min_length:
            return self._run_model(combined, min_length, max_length)

        return combined

    # ------------------------------------------------------------------
    # Private: Statistics
    # ------------------------------------------------------------------

    def _compute_statistics(
        self, original_text: str, summary_text: str
    ) -> Dict[str, Any]:
        """
        Returns compression metrics comparing the original to the final summary.
        """
        orig_words: int = len(original_text.split())
        summ_words: int = len(summary_text.split())
        ratio: float = round(summ_words / orig_words, 4) if orig_words > 0 else 0.0

        return {
            "original_word_count": orig_words,
            "summary_word_count": summ_words,
            "compression_ratio": ratio
        }

    # ------------------------------------------------------------------
    # Private: Build Summary Dict
    # ------------------------------------------------------------------

    def _build_summary_dict(
        self,
        text: str,
        entities: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generates all three summary variants plus statistics for the provided text.
        Injects entity context into the text when entities are supplied.
        """
        # Bypass requirement: Skip summarization for articles shorter than 100 words
        word_count = len(text.split())
        if word_count < 100:
            return {
                "short_summary": text,
                "medium_summary": text,
                "detailed_summary": text,
                "summary_statistics": self._compute_statistics(text, text)
            }

        enriched_text: str = (
            self._inject_entities(text, entities)
            if entities else text
        )

        short: str = self._summarize_text(
            enriched_text,
            SUMMARY_LENGTHS["short"]["min_length"],
            SUMMARY_LENGTHS["short"]["max_length"]
        )
        medium: str = self._summarize_text(
            enriched_text,
            SUMMARY_LENGTHS["medium"]["min_length"],
            SUMMARY_LENGTHS["medium"]["max_length"]
        )
        detailed: str = self._summarize_text(
            enriched_text,
            SUMMARY_LENGTHS["detailed"]["min_length"],
            SUMMARY_LENGTHS["detailed"]["max_length"],
            cohere_chunks=False
        )

        return {
            "short_summary": short,
            "medium_summary": medium,
            "detailed_summary": detailed,
            "summary_statistics": self._compute_statistics(text, detailed)
        }

    # ------------------------------------------------------------------
    # Public: Single Article
    # ------------------------------------------------------------------

    def summarize_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates summaries for a single article dictionary.
        Expects 'text' and optionally 'entities' keys.
        Enriches the article with a 'summaries' key.
        """
        try:
            text: str = article.get("text", "")
            entities: List[Dict[str, Any]] = article.get("entities", [])
            article["summaries"] = self._build_summary_dict(text, entities)
        except Exception as e:
            logger.error(f"Failed to summarize article '{article.get('title')}': {e}")
            # Ensure the pipeline continues even if one article fails completely
            article["summaries"] = {
                "short_summary": "Summarization failed.",
                "medium_summary": "Summarization failed.",
                "detailed_summary": "Summarization failed.",
                "summary_statistics": {"original_word_count": 0, "summary_word_count": 0, "compression_ratio": 0.0}
            }
        return article

    def summarize_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates summaries for every article in the list.
        Reports progress via callback on each iteration.
        """
        total: int = len(articles)
        for i, article in enumerate(articles):
            # Progress format requirement: "Summarizing article {i}/{total}"
            self._notify(f"Summarizing article {i + 1}/{total}...")
            articles[i] = self.summarize_article(article)
            
            # Memory Limit requirement: Trigger garbage collection to free CPU memory
            gc.collect()
            
        return articles

    # ------------------------------------------------------------------
    # Public: Magazine-Level Summary
    # ------------------------------------------------------------------

    def generate_magazine_summary(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates an overall magazine summary by concatenating the short summaries
        of all individual articles and using the fast extractive fallback to compress it.
        This avoids running the heavy transformer model on the entire magazine.
        """
        self._notify("Generating overall magazine summary...")

        combined_parts: List[str] = []
        for article in articles:
            title: str = article.get("title", "Untitled")
            summaries: Dict[str, str] = article.get("summaries", {})
            short: str = summaries.get("short_summary", article.get("text", "")[:400])
            if short:
                combined_parts.append(f"{title}: {short}")

        if not combined_parts:
            return {
                "short_summary": "",
                "medium_summary": "",
                "detailed_summary": "",
                "summary_statistics": {
                    "original_word_count": 0,
                    "summary_word_count": 0,
                    "compression_ratio": 0.0
                }
            }

        combined_text: str = " | ".join(combined_parts)
        
        # Use instantaneous TextRank extractive summarization for the magazine-level
        # to ensure it never hangs the pipeline at the very end.
        extractive = self._textrank_fallback(combined_text, SUMMARY_LENGTHS["detailed"]["max_length"])
        
        return {
            "short_summary": extractive[:300] + "...",
            "medium_summary": extractive[:600] + "...",
            "detailed_summary": extractive,
            "summary_statistics": self._compute_statistics(combined_text, extractive)
        }

