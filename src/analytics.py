import logging
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """
    Aggregates the outputs of all NLP modules (keywords, entities, topics,
    sentiment, summaries) into one structured analytics report.
    All computation is performed locally with no external calls.
    """

    def __init__(self, articles: List[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
        """
        Args:
            articles: List of fully enriched article dictionaries produced by
                      all upstream modules (detector, keywords, entities, topics,
                      sentiment, summarizer).
            metadata: Magazine-level metadata dict from PDFExtractor, containing
                      at minimum 'page_count' and 'title'.
        """
        self._articles: List[Dict[str, Any]] = articles if articles else []
        self._metadata: Dict[str, Any] = metadata if metadata else {}

    # ------------------------------------------------------------------
    # Private: Magazine Statistics
    # ------------------------------------------------------------------

    def _compute_magazine_statistics(self) -> Dict[str, Any]:
        """
        Returns high-level magazine statistics derived from all articles.
        """
        total_articles: int = len(self._articles)
        total_pages: int = int(self._metadata.get("page_count", 0))

        word_counts: List[int] = [
            a.get("word_count", 0) for a in self._articles
        ]
        total_words: int = sum(word_counts)
        avg_article_length: float = round(
            total_words / total_articles, 1
        ) if total_articles > 0 else 0.0

        reading_times: List[int] = [
            a.get("reading_time", 0) for a in self._articles
        ]
        total_reading_time: int = sum(reading_times)

        all_words: List[str] = []
        for article in self._articles:
            all_words.extend(article.get("text", "").lower().split())

        unique_words: int = len(set(all_words))
        vocabulary_richness: float = round(
            unique_words / len(all_words), 4
        ) if all_words else 0.0

        return {
            "total_pages": total_pages,
            "total_articles": total_articles,
            "total_words": total_words,
            "average_article_length": avg_article_length,
            "total_reading_time_seconds": total_reading_time,
            "total_reading_time_minutes": round(total_reading_time / 60, 1),
            "vocabulary_richness": vocabulary_richness,
            "unique_words": unique_words
        }

    # ------------------------------------------------------------------
    # Private: Keyword Analytics
    # ------------------------------------------------------------------

    def _compute_keyword_analytics(self) -> Dict[str, Any]:
        """
        Aggregates keyword data across all articles.
        """
        frequency: Counter = Counter()
        all_keyword_sets: List[int] = []

        for article in self._articles:
            keywords: List[Dict[str, Any]] = article.get("keywords", [])
            all_keyword_sets.append(len(keywords))
            for kw in keywords:
                phrase: str = kw.get("keyword", "").lower().strip()
                if phrase:
                    frequency[phrase] += 1

        top_keywords: List[Dict[str, Any]] = [
            {"keyword": kw, "frequency": count}
            for kw, count in frequency.most_common(20)
        ]

        total_unique_keywords: int = len(frequency)
        avg_keywords_per_article: float = round(
            sum(all_keyword_sets) / len(all_keyword_sets), 1
        ) if all_keyword_sets else 0.0

        return {
            "top_keywords": top_keywords,
            "keyword_frequency": dict(frequency.most_common(50)),
            "keyword_diversity": total_unique_keywords,
            "average_keywords_per_article": avg_keywords_per_article
        }

    # ------------------------------------------------------------------
    # Private: Entity Analytics
    # ------------------------------------------------------------------

    def _compute_entity_analytics(self) -> Dict[str, Any]:
        """
        Aggregates named entity data across all articles.
        """
        label_counters: Dict[str, Counter] = {
            "PERSON": Counter(),
            "ORG":    Counter(),
            "GPE":    Counter(),
            "LOC":    Counter(),
        }
        entity_counts_per_article: List[int] = []

        for article in self._articles:
            entities: List[Dict[str, Any]] = article.get("entities", [])
            entity_counts_per_article.append(len(entities))
            for ent in entities:
                label: str = ent.get("label", "")
                text: str = ent.get("text", "").strip()
                if label in label_counters and text:
                    label_counters[label][text] += 1

        def top_entities(counter: Counter, n: int = 10) -> List[Dict[str, Any]]:
            return [
                {"entity": name, "count": count}
                for name, count in counter.most_common(n)
            ]

        location_counter: Counter = label_counters["GPE"] + label_counters["LOC"]
        avg_entities: float = round(
            sum(entity_counts_per_article) / len(entity_counts_per_article), 1
        ) if entity_counts_per_article else 0.0

        return {
            "most_common_people": top_entities(label_counters["PERSON"]),
            "most_common_organizations": top_entities(label_counters["ORG"]),
            "most_common_locations": top_entities(location_counter),
            "average_entities_per_article": avg_entities
        }

    # ------------------------------------------------------------------
    # Private: Topic Analytics
    # ------------------------------------------------------------------

    def _compute_topic_analytics(self) -> Dict[str, Any]:
        """
        Aggregates topic classification results across all articles.
        """
        label_counts: Counter = Counter()
        confidence_sum: float = 0.0
        valid_count: int = 0

        for article in self._articles:
            category: Dict[str, Any] = article.get("category", {})
            label: str = category.get("label", "Unknown")
            confidence: float = float(category.get("confidence", 0.0))
            label_counts[label] += 1
            confidence_sum += confidence
            valid_count += 1

        total: int = sum(label_counts.values())
        topic_distribution: Dict[str, float] = {
            label: round((count / total) * 100, 2)
            for label, count in label_counts.items()
        } if total > 0 else {}

        dominant_topic: str = (
            label_counts.most_common(1)[0][0] if label_counts else "Unknown"
        )
        avg_confidence: float = round(
            confidence_sum / valid_count, 4
        ) if valid_count > 0 else 0.0

        return {
            "topic_distribution": topic_distribution,
            "dominant_topic": dominant_topic,
            "topic_frequency": dict(label_counts.most_common()),
            "average_topic_confidence": avg_confidence
        }

    # ------------------------------------------------------------------
    # Private: Sentiment Analytics
    # ------------------------------------------------------------------

    def _compute_sentiment_analytics(self) -> Dict[str, Any]:
        """
        Aggregates sentiment data across all articles.
        """
        label_counts: Counter = Counter()
        confidence_sum: float = 0.0
        valid_count: int = 0

        for article in self._articles:
            sentiment: Dict[str, Any] = article.get("sentiment", {})
            label: str = sentiment.get("label", "NEUTRAL")
            confidence: float = float(sentiment.get("confidence", 0.0))
            label_counts[label] += 1
            confidence_sum += confidence
            valid_count += 1

        total: int = sum(label_counts.values())

        positive_pct: float = round(
            (label_counts.get("POSITIVE", 0) / total) * 100, 2
        ) if total > 0 else 0.0
        neutral_pct: float = round(
            (label_counts.get("NEUTRAL", 0) / total) * 100, 2
        ) if total > 0 else 0.0
        negative_pct: float = round(
            (label_counts.get("NEGATIVE", 0) / total) * 100, 2
        ) if total > 0 else 0.0

        avg_confidence: float = round(
            confidence_sum / valid_count, 4
        ) if valid_count > 0 else 0.0

        return {
            "positive_percent": positive_pct,
            "neutral_percent": neutral_pct,
            "negative_percent": negative_pct,
            "average_sentiment_confidence": avg_confidence,
            "dominant_sentiment": (
                label_counts.most_common(1)[0][0] if label_counts else "NEUTRAL"
            )
        }

    # ------------------------------------------------------------------
    # Private: Reading Analytics
    # ------------------------------------------------------------------

    def _compute_reading_analytics(self) -> Dict[str, Any]:
        """
        Derives per-article reading metrics and identifies extremes.
        """
        if not self._articles:
            return {
                "longest_article": None,
                "shortest_article": None,
                "fastest_read": None,
                "slowest_read": None
            }

        by_words: List[Dict[str, Any]] = sorted(
            self._articles, key=lambda a: a.get("word_count", 0)
        )
        by_time: List[Dict[str, Any]] = sorted(
            self._articles, key=lambda a: a.get("reading_time", 0)
        )

        def article_summary(article: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "title": article.get("title", "Untitled"),
                "word_count": article.get("word_count", 0),
                "reading_time_seconds": article.get("reading_time", 0)
            }

        return {
            "longest_article": article_summary(by_words[-1]),
            "shortest_article": article_summary(by_words[0]),
            "fastest_read": article_summary(by_time[0]),
            "slowest_read": article_summary(by_time[-1])
        }

    # ------------------------------------------------------------------
    # Private: Quality Metrics
    # ------------------------------------------------------------------

    def _compute_quality_metrics(self) -> Dict[str, Any]:
        """
        Computes average quality and compression metrics across all articles.
        """
        compression_ratios: List[float] = []
        entity_counts: List[int] = []
        keyword_counts: List[int] = []

        for article in self._articles:
            summaries: Dict[str, Any] = article.get("summaries", {})
            stats: Dict[str, Any] = summaries.get("summary_statistics", {})
            ratio: Optional[float] = stats.get("compression_ratio")
            if ratio is not None:
                compression_ratios.append(float(ratio))

            entity_counts.append(len(article.get("entities", [])))
            keyword_counts.append(len(article.get("keywords", [])))

        def safe_avg(values: List[float]) -> float:
            return round(sum(values) / len(values), 4) if values else 0.0

        return {
            "average_summary_compression": safe_avg(compression_ratios),
            "average_entity_count": safe_avg(entity_counts),
            "average_keyword_count": safe_avg(keyword_counts)
        }

    # ------------------------------------------------------------------
    # Public: Full Report
    # ------------------------------------------------------------------

    def generate_report(self) -> Dict[str, Any]:
        """
        Runs all analytics computations and returns a single structured report.

        Returns:
            Dict containing all 7 analytics sections:
            - magazine_statistics
            - keyword_analytics
            - entity_analytics
            - topic_analytics
            - sentiment_analytics
            - reading_analytics
            - quality_metrics
        """
        logger.info("Generating full analytics report...")

        return {
            "magazine_statistics": self._compute_magazine_statistics(),
            "keyword_analytics":   self._compute_keyword_analytics(),
            "entity_analytics":    self._compute_entity_analytics(),
            "topic_analytics":     self._compute_topic_analytics(),
            "sentiment_analytics": self._compute_sentiment_analytics(),
            "reading_analytics":   self._compute_reading_analytics(),
            "quality_metrics":     self._compute_quality_metrics()
        }
