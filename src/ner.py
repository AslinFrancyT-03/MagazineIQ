import logging
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

import spacy
from spacy.language import Language

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_LABELS: Set[str] = {
    "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "DATE", "MONEY", "NORP"
}

SPACY_MODEL: str = "en_core_web_sm"


class EntityExtractor:
    """
    Extracts named entities from article text using spaCy.
    Supports single-article extraction and magazine-level entity aggregation.
    """

    def __init__(self) -> None:
        self._nlp: Language = self._load_model()

    def _load_model(self) -> Language:
        """
        Loads the spaCy model. Falls back gracefully if the model is not installed.
        """
        try:
            nlp: Language = spacy.load(SPACY_MODEL)
            logger.info(f"spaCy model '{SPACY_MODEL}' loaded successfully.")
            return nlp
        except OSError:
            logger.error(
                f"spaCy model '{SPACY_MODEL}' not found. "
                f"Run: python -m spacy download {SPACY_MODEL}"
            )
            raise RuntimeError(
                f"Required spaCy model '{SPACY_MODEL}' is not installed. "
                f"Please run: python -m spacy download {SPACY_MODEL}"
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Runs spaCy NER on raw text and returns a deduplicated list of entity dicts.
        Only entities with labels in SUPPORTED_LABELS are retained.

        Returns:
            List of dicts: [{"text": str, "label": str, "start_char": int, "end_char": int}]
        """
        if not text or not text.strip():
            return []

        doc = self._nlp(text)

        seen: Set[Tuple[str, str]] = set()
        entities: List[Dict[str, Any]] = []

        for ent in doc.ents:
            label: str = ent.label_
            entity_text: str = ent.text.strip()

            if label not in SUPPORTED_LABELS:
                continue

            if not entity_text:
                continue

            # Deduplicate by normalised (text, label) pair within a single article
            dedup_key: Tuple[str, str] = (entity_text.lower(), label)
            if dedup_key in seen:
                continue

            seen.add(dedup_key)
            entities.append({
                "text": entity_text,
                "label": label,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })

        return entities

    # ------------------------------------------------------------------
    # Public: Single Article
    # ------------------------------------------------------------------

    def extract_from_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts named entities from a single article dictionary.
        Expects the article to contain a 'text' key.
        Enriches the article with an 'entities' key.

        Returns:
            The article dict enriched with 'entities'.
        """
        text: str = article.get("text", "")
        article["entities"] = self._extract_from_text(text)
        return article

    def extract_from_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts named entities from every article in the provided list.

        Returns:
            The same list with each article enriched with an 'entities' key.
        """
        return [self.extract_from_article(article) for article in articles]

    # ------------------------------------------------------------------
    # Public: Magazine-Level Aggregation
    # ------------------------------------------------------------------

    def aggregate_magazine_entities(
        self, articles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregates entity data across all articles to produce magazine-level insights.

        Returns:
            Dict with:
              - most_mentioned_people (List[Dict]): Top 10 PERSON entities by frequency.
              - most_mentioned_organizations (List[Dict]): Top 10 ORG entities by frequency.
              - most_mentioned_locations (List[Dict]): Top 10 GPE + LOC entities by frequency.
              - entity_frequency (Dict[str, int]): Full frequency count for every entity.
              - entity_statistics (Dict): Counts per label across the magazine.
        """
        label_counters: Dict[str, Counter] = {label: Counter() for label in SUPPORTED_LABELS}
        global_frequency: Counter = Counter()

        for article in articles:
            entities: List[Dict[str, Any]] = article.get("entities", [])
            for ent in entities:
                text: str = ent.get("text", "").strip()
                label: str = ent.get("label", "")
                if text and label in SUPPORTED_LABELS:
                    label_counters[label][text] += 1
                    global_frequency[text] += 1

        def to_ranked_list(counter: Counter, top_n: int = 10) -> List[Dict[str, Any]]:
            return [
                {"entity": entity, "count": count}
                for entity, count in counter.most_common(top_n)
            ]

        # Merge GPE and LOC for location ranking
        location_counter: Counter = label_counters["GPE"] + label_counters["LOC"]

        entity_statistics: Dict[str, int] = {
            label: sum(label_counters[label].values())
            for label in SUPPORTED_LABELS
        }

        return {
            "most_mentioned_people": to_ranked_list(label_counters["PERSON"]),
            "most_mentioned_organizations": to_ranked_list(label_counters["ORG"]),
            "most_mentioned_locations": to_ranked_list(location_counter),
            "entity_frequency": dict(global_frequency.most_common()),
            "entity_statistics": entity_statistics
        }
