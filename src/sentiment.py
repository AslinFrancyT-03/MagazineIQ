import logging
from collections import Counter
from typing import Any, Dict, List, Optional

from transformers import pipeline, Pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL: str = "distilbert-base-uncased-finetuned-sst-2-english"

# Maximum characters per chunk sent to the model
MAX_CHARS_PER_CHUNK: int = 512

# Mapping model-native labels to normalised three-class labels
LABEL_MAP: Dict[str, str] = {
    "POSITIVE": "POSITIVE",
    "NEGATIVE": "NEGATIVE",
    "NEUTRAL":  "NEUTRAL",
    "LABEL_0":  "NEGATIVE",
    "LABEL_1":  "NEUTRAL",
    "LABEL_2":  "POSITIVE",
}


class SentimentAnalyzer:
    """
    Performs sentiment analysis on individual articles and across an entire magazine
    using a fully local HuggingFace transformer pipeline.
    Long articles are automatically chunked and their results are aggregated.
    """

    _cached_pipeline: Optional[Pipeline] = None
    _cached_model_name: Optional[str] = None

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name: str = model_name
        self._pipeline: Pipeline = self._load_pipeline()

    # ------------------------------------------------------------------
    # Private: Model Loading (with class-level cache)
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> Pipeline:
        """
        Loads the sentiment-analysis pipeline.
        Uses a class-level cache so the model is loaded only once per process.
        """
        if (
            SentimentAnalyzer._cached_pipeline is not None
            and SentimentAnalyzer._cached_model_name == self._model_name
        ):
            logger.info(f"Using cached sentiment model: {self._model_name}")
            return SentimentAnalyzer._cached_pipeline

        try:
            logger.info(f"Loading sentiment model: {self._model_name}")
            loaded: Pipeline = pipeline(
                "sentiment-analysis",
                model=self._model_name,
                tokenizer=self._model_name,
                device=-1,          # CPU; set to 0 for GPU
                truncation=True,
                max_length=512
            )
            SentimentAnalyzer._cached_pipeline = loaded
            SentimentAnalyzer._cached_model_name = self._model_name
            logger.info(f"Sentiment model '{self._model_name}' loaded and cached.")
            return loaded
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            raise RuntimeError(
                f"Could not load sentiment model '{self._model_name}'. "
                "Ensure transformers is installed and the model is available."
            ) from e

    # ------------------------------------------------------------------
    # Private: Chunking
    # ------------------------------------------------------------------

    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Splits text into character-bounded chunks that each fit within the
        model's token limit.
        """
        chunks: List[str] = []
        for i in range(0, len(text), MAX_CHARS_PER_CHUNK):
            chunk: str = text[i: i + MAX_CHARS_PER_CHUNK].strip()
            if chunk:
                chunks.append(chunk)
        return chunks if chunks else [text]

    # ------------------------------------------------------------------
    # Private: Label Normalisation & Aggregation
    # ------------------------------------------------------------------

    def _normalise_label(self, raw_label: str) -> str:
        """Maps model-specific label strings to POSITIVE / NEGATIVE / NEUTRAL."""
        return LABEL_MAP.get(raw_label.upper(), "NEUTRAL")

    def _aggregate_chunk_results(
        self, chunk_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Combines sentiment results from multiple chunks into a single final result.
        Strategy: weighted vote by confidence score.
        The label with the highest total accumulated confidence wins.
        """
        label_scores: Dict[str, float] = {"POSITIVE": 0.0, "NEGATIVE": 0.0, "NEUTRAL": 0.0}
        chunk_count: int = len(chunk_results)

        for chunk in chunk_results:
            label: str = self._normalise_label(chunk.get("label", "NEUTRAL"))
            score: float = float(chunk.get("score", 0.0))
            label_scores[label] += score

        winning_label: str = max(label_scores, key=lambda k: label_scores[k])
        avg_confidence: float = round(
            label_scores[winning_label] / chunk_count, 4
        ) if chunk_count > 0 else 0.0

        return {"label": winning_label, "confidence": avg_confidence}

    # ------------------------------------------------------------------
    # Private: Core Analysis
    # ------------------------------------------------------------------

    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyzes sentiment of a single text block.
        Handles long text via chunking and aggregation.

        Returns:
            Dict: {"label": str, "confidence": float}
        """
        if not text or not text.strip():
            return {"label": "NEUTRAL", "confidence": 0.0}

        chunks: List[str] = self._split_into_chunks(text)
        chunk_results: List[Dict[str, Any]] = []

        for chunk in chunks:
            try:
                result: List[Dict[str, Any]] = self._pipeline(chunk)
                if result:
                    chunk_results.append(result[0])
            except Exception as e:
                logger.warning(f"Chunk sentiment inference failed: {e}")

        if not chunk_results:
            return {"label": "NEUTRAL", "confidence": 0.0}

        if len(chunk_results) == 1:
            raw: Dict[str, Any] = chunk_results[0]
            return {
                "label": self._normalise_label(raw.get("label", "NEUTRAL")),
                "confidence": round(float(raw.get("score", 0.0)), 4)
            }

        return self._aggregate_chunk_results(chunk_results)

    # ------------------------------------------------------------------
    # Public: Single Article
    # ------------------------------------------------------------------

    def analyze_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs sentiment analysis on a single article dictionary.
        Expects a 'text' key. Enriches the article with a 'sentiment' key.

        Returns:
            The article dict enriched with 'sentiment'.
        """
        text: str = article.get("text", "")
        article["sentiment"] = self._analyze_text(text)
        return article

    def analyze_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Performs sentiment analysis on every article in the provided list.

        Returns:
            The same list with each article enriched with a 'sentiment' key.
        """
        total: int = len(articles)
        for i, article in enumerate(articles):
            title: str = article.get("title", "Untitled")
            logger.info(f"Analyzing sentiment {i + 1}/{total}: {title}")
            articles[i] = self.analyze_article(article)
        return articles

    # ------------------------------------------------------------------
    # Public: Magazine-Level Statistics
    # ------------------------------------------------------------------

    def aggregate_magazine_sentiment(
        self, articles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregates sentiment data across all articles.

        Returns:
            Dict with:
              - sentiment_distribution (Dict[str, float]): Percentage per label.
              - average_sentiment_confidence (float): Mean confidence across articles.
              - dominant_sentiment (str): Most frequent sentiment label.
        """
        label_counts: Counter = Counter()
        confidence_sum: float = 0.0
        valid_count: int = 0

        for article in articles:
            sentiment: Dict[str, Any] = article.get("sentiment", {})
            label: str = sentiment.get("label", "NEUTRAL")
            confidence: float = float(sentiment.get("confidence", 0.0))

            label_counts[label] += 1
            confidence_sum += confidence
            valid_count += 1

        total: int = sum(label_counts.values())

        sentiment_distribution: Dict[str, float] = {
            label: round((count / total) * 100, 2)
            for label, count in label_counts.items()
        } if total > 0 else {}

        average_confidence: float = round(
            confidence_sum / valid_count, 4
        ) if valid_count > 0 else 0.0

        dominant_sentiment: str = (
            label_counts.most_common(1)[0][0] if label_counts else "NEUTRAL"
        )

        return {
            "sentiment_distribution": sentiment_distribution,
            "average_sentiment_confidence": average_confidence,
            "dominant_sentiment": dominant_sentiment
        }
