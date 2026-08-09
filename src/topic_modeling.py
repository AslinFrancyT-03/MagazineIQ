import logging
from collections import Counter
from typing import Any, Dict, List, Optional

from transformers import pipeline, Pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL: str = "typeform/distilbert-base-uncased-mnli"

CANDIDATE_LABELS: List[str] = [
    "Technology", "Business", "Finance", "Health", "Science",
    "Sports", "Politics", "Education", "Entertainment",
    "Lifestyle", "Travel", "Environment"
]

# Maximum characters of article text to send to the model.
# Zero-shot classification does not require the full text — the first
# 1500 characters carry the most topic-relevant signal.
MAX_INPUT_CHARS: int = 1500


class TopicClassifier:
    """
    Classifies each article into one of the predefined magazine topic categories
    using a local zero-shot classification transformer model.
    The model is loaded once and cached for the lifetime of the instance.
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
        Loads the zero-shot classification pipeline.
        Uses a class-level cache so the model is only loaded once per process,
        even if multiple TopicClassifier instances are created.
        """
        if (
            TopicClassifier._cached_pipeline is not None
            and TopicClassifier._cached_model_name == self._model_name
        ):
            logger.info(f"Using cached classifier model: {self._model_name}")
            return TopicClassifier._cached_pipeline

        try:
            logger.info(f"Loading zero-shot classifier: {self._model_name}")
            loaded: Pipeline = pipeline(
                "zero-shot-classification",
                model=self._model_name,
                device=-1  # CPU; change to 0 to use GPU
            )
            TopicClassifier._cached_pipeline = loaded
            TopicClassifier._cached_model_name = self._model_name
            logger.info(f"Model '{self._model_name}' loaded and cached.")
            return loaded
        except Exception as e:
            logger.error(f"Failed to load topic classifier model: {e}")
            raise RuntimeError(
                f"Could not load zero-shot classification model '{self._model_name}'. "
                "Ensure transformers is installed and the model is downloaded."
            ) from e

    # ------------------------------------------------------------------
    # Private: Classification Logic
    # ------------------------------------------------------------------

    def _classify_text(self, text: str) -> Dict[str, Any]:
        """
        Runs zero-shot classification on the provided text.
        Truncates the input to MAX_INPUT_CHARS to stay within token limits.

        Returns:
            Dict with 'label' (str) and 'confidence' (float).
        """
        if not text or not text.strip():
            return {"label": "Unknown", "confidence": 0.0}

        truncated: str = text.strip()[:MAX_INPUT_CHARS]

        try:
            result: Dict[str, Any] = self._pipeline(
                truncated,
                candidate_labels=CANDIDATE_LABELS,
                multi_label=False
            )
            top_label: str = result["labels"][0]
            top_score: float = round(float(result["scores"][0]), 4)
            return {"label": top_label, "confidence": top_score}
        except Exception as e:
            logger.warning(f"Classification inference failed: {e}")
            return {"label": "Unknown", "confidence": 0.0}

    # ------------------------------------------------------------------
    # Public: Single Article
    # ------------------------------------------------------------------

    def classify_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classifies a single article dictionary.
        Expects the article to contain a 'text' key.
        Enriches the article with a 'category' key.

        Returns:
            The article dict enriched with 'category'.
        """
        text: str = article.get("text", "")
        article["category"] = self._classify_text(text)
        return article

    def classify_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classifies every article in the provided list.

        Returns:
            The same list with each article enriched with a 'category' key.
        """
        total: int = len(articles)
        for i, article in enumerate(articles):
            title: str = article.get("title", "Untitled")
            logger.info(f"Classifying article {i + 1}/{total}: {title}")
            articles[i] = self.classify_article(article)
        return articles

    # ------------------------------------------------------------------
    # Public: Magazine-Level Statistics
    # ------------------------------------------------------------------

    def aggregate_magazine_topics(
        self, articles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregates topic classification results across all articles.

        Returns:
            Dict with:
              - topic_distribution (Dict[str, float]): Percentage share per topic.
              - dominant_topic (str): The most frequently assigned topic.
              - topic_frequency (Dict[str, int]): Raw article count per topic.
        """
        label_counts: Counter = Counter()

        for article in articles:
            category: Dict[str, Any] = article.get("category", {})
            label: str = category.get("label", "Unknown")
            label_counts[label] += 1

        total: int = sum(label_counts.values())

        topic_frequency: Dict[str, int] = dict(label_counts.most_common())

        topic_distribution: Dict[str, float] = {
            label: round((count / total) * 100, 2)
            for label, count in label_counts.items()
        } if total > 0 else {}

        dominant_topic: str = (
            label_counts.most_common(1)[0][0] if label_counts else "Unknown"
        )

        return {
            "topic_distribution": topic_distribution,
            "dominant_topic": dominant_topic,
            "topic_frequency": topic_frequency
        }
