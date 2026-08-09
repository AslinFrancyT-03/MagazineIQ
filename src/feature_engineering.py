import yake
import logging
from collections import Counter
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOP_N: int = 15
MAX_NGRAM_SIZE: int = 3
DEDUPLICATION_THRESHOLD: float = 0.9
LANGUAGE: str = "en"

STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "was", "are", "were",
    "be", "been", "has", "have", "had", "it", "its", "this", "that",
    "these", "those", "as", "into", "about", "than", "so", "also",
    "which", "who", "what", "when", "where", "how", "not", "no", "if"
}


class KeywordExtractor:
    """
    Extracts meaningful keyword phrases from article text using YAKE.
    Supports both single-article extraction and magazine-level aggregation.
    """

    def __init__(self) -> None:
        self._extractor: yake.KeywordExtractor = yake.KeywordExtractor(
            lan=LANGUAGE,
            n=MAX_NGRAM_SIZE,
            dedupLim=DEDUPLICATION_THRESHOLD,
            top=TOP_N,
            features=None
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_stopword_only(self, phrase: str) -> bool:
        """Returns True if every token in the phrase is a stopword."""
        tokens: List[str] = phrase.lower().split()
        return all(token in STOPWORDS for token in tokens)

    def _deduplicate(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Removes duplicate keyword entries by normalising to lowercase.
        Keeps the entry with the lowest (best) YAKE score when a duplicate is found.
        """
        seen: Dict[str, float] = {}
        for item in keywords:
            key: str = item["keyword"].lower().strip()
            if key not in seen or item["score"] < seen[key]:
                seen[key] = item["score"]

        return [
            {"keyword": k, "score": round(v, 6)}
            for k, v in sorted(seen.items(), key=lambda x: x[1])
        ]

    def _extract_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Runs YAKE on a single text string.
        Filters stopword-only phrases and deduplicates results.

        Returns:
            List of dicts: [{"keyword": str, "score": float}, ...]
        """
        if not text or not text.strip():
            return []

        try:
            raw_keywords: List[tuple] = self._extractor.extract_keywords(text)
        except Exception as e:
            logger.error(f"YAKE extraction error: {e}")
            return []

        cleaned: List[Dict[str, Any]] = []
        for phrase, score in raw_keywords:
            if phrase and not self._is_stopword_only(phrase):
                cleaned.append({"keyword": phrase.strip(), "score": score})

        return self._deduplicate(cleaned)

    # ------------------------------------------------------------------
    # Public: Single Article Extraction
    # ------------------------------------------------------------------

    def extract_from_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts keywords from a single article dictionary.
        Expects the article to contain a 'text' key.

        Returns:
            The article dict enriched with a 'keywords' key.
        """
        text: str = article.get("text", "")
        keywords: List[Dict[str, Any]] = self._extract_from_text(text)
        article["keywords"] = keywords
        return article

    def extract_from_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts keywords for every article in the provided list.

        Returns:
            The same list with each article enriched with a 'keywords' key.
        """
        return [self.extract_from_article(article) for article in articles]

    # ------------------------------------------------------------------
    # Public: Magazine-Level Aggregation
    # ------------------------------------------------------------------

    def aggregate_magazine_keywords(
        self, articles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregates keyword data across all articles to produce magazine-level insights.

        Returns:
            Dict with:
              - top_keywords (List[Dict]): Top 20 keywords by frequency and importance.
              - keyword_frequency (Dict[str, int]): How many articles mention each keyword.
              - keyword_importance (Dict[str, float]): Best (lowest) YAKE score per keyword.
        """
        frequency: Counter = Counter()
        importance: Dict[str, float] = {}

        for article in articles:
            keywords: List[Dict[str, Any]] = article.get("keywords", [])
            seen_in_article: Set[str] = set()

            for item in keywords:
                phrase: str = item["keyword"].lower().strip()
                score: float = item["score"]

                if phrase not in seen_in_article:
                    frequency[phrase] += 1
                    seen_in_article.add(phrase)

                # Keep the best (lowest) YAKE score for importance ranking
                if phrase not in importance or score < importance[phrase]:
                    importance[phrase] = score

        # Build top keywords ranked by frequency first, then importance
        all_phrases: List[str] = [kw for kw, _ in frequency.most_common()]
        top_keywords: List[Dict[str, Any]] = [
            {
                "keyword": phrase,
                "frequency": frequency[phrase],
                "importance_score": round(importance.get(phrase, 1.0), 6)
            }
            for phrase in all_phrases[:20]
        ]

        return {
            "top_keywords": top_keywords,
            "keyword_frequency": dict(frequency.most_common()),
            "keyword_importance": {
                k: round(v, 6)
                for k, v in sorted(importance.items(), key=lambda x: x[1])[:20]
            }
        }
