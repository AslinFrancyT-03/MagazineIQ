import re
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORDS_PER_MINUTE: int = 200

TOC_KEYWORDS: set = {
    "table of contents", "contents", "in this issue",
    "this month", "what's inside", "index"
}

SECTION_HEADINGS: set = {
    "feature", "opinion", "analysis", "interview", "review",
    "technology", "business", "lifestyle", "science", "health",
    "sports", "culture", "editorial", "perspective", "focus",
    "special report", "world", "economy"
}

AD_PATTERNS: List[re.Pattern] = [
    re.compile(r"(advertis(ement|ing)|sponsored|paid\s+content)", re.IGNORECASE),
    re.compile(r"\b(offer|discount|sale|promo|coupon)\b", re.IGNORECASE),
    re.compile(r"\d{1,3}\s*%\s*off", re.IGNORECASE),
    re.compile(r"(call\s+now|buy\s+now|order\s+today|subscribe\s+now)", re.IGNORECASE),
    re.compile(r"(www\.|https?://)", re.IGNORECASE),
    re.compile(r"\+?\d[\d\s\-]{8,}", re.IGNORECASE),
]

CONTINUATION_PATTERNS: List[re.Pattern] = [
    re.compile(r"\(continued\)", re.IGNORECASE),
    re.compile(r"continued\s+from\s+page\s+\d+", re.IGNORECASE),
    re.compile(r"continued\s+on\s+page\s+\d+", re.IGNORECASE),
    re.compile(r"^\s*cont(\.|\s*d\.?)\s*$", re.IGNORECASE),
]


class ArticleDetector:

    def __init__(self, pages: List[Dict[str, Any]]) -> None:
        self.pages: List[Dict[str, Any]] = pages if pages else []

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _is_table_of_contents(self, text: str) -> bool:
        sample: str = text[:300].lower()
        for keyword in TOC_KEYWORDS:
            if keyword in sample:
                return True
        return len(re.findall(r"\.\s*\.\s*\.\s*\d+", text)) >= 3

    def _is_advertisement(self, text: str) -> bool:
        if len(text.split()) < 5:
            return False
        for pattern in AD_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def _is_section_heading(self, line: str) -> bool:
        return line.strip().lower() in SECTION_HEADINGS

    def _is_continuation_page(self, text: str) -> bool:
        """
        Returns True if the page or block starts with a continuation marker,
        meaning its content should be merged into the previously detected article.
        Checks the first 200 characters only to avoid false positives inside body text.
        """
        sample: str = text[:200]
        for pattern in CONTINUATION_PATTERNS:
            if pattern.search(sample):
                return True
        return False

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_title_candidate(self, line: str) -> Dict[str, Any]:
        """
        Multi-heuristic title scoring.

        Returns:
            Dict with keys:
              - score (float): confidence value in [0.0, 1.0]
              - reason (str): human-readable explanation of the score
        """
        clean: str = line.strip()
        if not clean:
            return {"score": 0.0, "reason": "Empty line."}

        words: List[str] = clean.split()
        word_count: int = len(words)

        if word_count < 2 or word_count > 15:
            return {
                "score": 0.0,
                "reason": f"Word count {word_count} outside title range (2–15)."
            }

        score: float = 0.0
        reasons: List[str] = []

        if 2 <= word_count <= 12:
            score += 0.2
            reasons.append("Word count in optimal title range.")

        if clean[-1] not in {".", ",", ";", ":", "!", "?"}:
            score += 0.2
            reasons.append("No sentence-closing punctuation.")

        if clean.isupper():
            score += 0.3
            reasons.append("All uppercase.")

        if clean.istitle():
            score += 0.2
            reasons.append("Title case detected.")

        caps_ratio: float = sum(1 for c in clean if c.isupper()) / len(clean)
        if caps_ratio > 0.4:
            score += 0.15
            reasons.append(f"High capital ratio ({caps_ratio:.0%}).")

        if re.search(r"(https?://|www\.|\d{4,})", clean):
            score -= 0.3
            reasons.append("Contains URL or long number — likely body text.")

        body_starters: set = {"the", "a", "an", "in", "on", "at", "by", "for", "with"}
        if words[0].lower() in body_starters:
            score -= 0.1
            reasons.append("Starts with a common body-text word.")

        final_score: float = round(min(max(score, 0.0), 1.0), 3)
        return {
            "score": final_score,
            "reason": " ".join(reasons) if reasons else "No strong signals detected."
        }

    def _is_potential_title(self, line: str) -> bool:
        if self._is_section_heading(line):
            return False
        result: Dict[str, Any] = self._score_title_candidate(line)
        return result["score"] >= 0.5

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------

    def _compute_reading_time(self, text: str) -> int:
        """Returns estimated reading time in seconds."""
        word_count: int = len(text.split())
        return max(1, round((word_count / WORDS_PER_MINUTE) * 60))

    def _enrich_article(
        self,
        article: Dict[str, Any],
        confidence_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Injects derived metadata into an article dictionary.
        Confidence is stored as {"score": float, "reason": str}.
        """
        text: str = article.get("text", "")

        if confidence_results:
            avg_score: float = round(
                sum(r["score"] for r in confidence_results) / len(confidence_results), 3
            )
            combined_reason: str = "; ".join(
                set(r["reason"] for r in confidence_results if r["reason"])
            )
        else:
            avg_score = 0.5
            combined_reason = "Default confidence."

        article["article_id"] = str(uuid.uuid4())
        article["word_count"] = len(text.split())
        article["character_count"] = len(text)
        article["reading_time"] = self._compute_reading_time(text)
        article["confidence"] = {
            "score": avg_score,
            "reason": combined_reason
        }
        return article

    # ------------------------------------------------------------------
    # Fallback & Statistics
    # ------------------------------------------------------------------

    def _apply_fallback(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        If no articles were detected, combine all pages into one fallback article.
        """
        if articles:
            return articles

        logger.info("No articles detected. Applying full-magazine fallback.")
        if not self.pages:
            return []

        full_text: str = "\n".join(p.get("text", "") for p in self.pages)
        fallback: Dict[str, Any] = {
            "title": "Main Content",
            "section": "General",
            "start_page": self.pages[0].get("page_number", 1),
            "end_page": self.pages[-1].get("page_number", 1),
            "text": full_text,
        }
        return [self._enrich_article(fallback, [{"score": 0.2, "reason": "Fallback article."}])]

    def _compute_statistics(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Computes magazine-level statistics across all detected articles.

        Returns:
            Dict with total_articles, average_words, largest_article, smallest_article.
        """
        if not articles:
            return {
                "total_articles": 0,
                "average_words": 0,
                "largest_article": None,
                "smallest_article": None
            }

        word_counts: List[Tuple[str, int]] = [
            (a.get("title", "Untitled"), a.get("word_count", 0)) for a in articles
        ]
        total: int = len(articles)
        avg: float = round(sum(wc for _, wc in word_counts) / total, 1)
        largest: str = max(word_counts, key=lambda x: x[1])[0]
        smallest: str = min(word_counts, key=lambda x: x[1])[0]

        return {
            "total_articles": total,
            "average_words": avg,
            "largest_article": largest,
            "smallest_article": smallest
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def detect(self) -> Dict[str, Any]:
        """
        Executes the full detection pipeline and returns articles plus statistics.

        Returns:
            Dict with:
              - articles (List[Dict]): Enriched article records.
              - statistics (Dict): Magazine-level aggregate stats.
        """
        if not self.pages:
            return {"articles": [], "statistics": self._compute_statistics([])}

        raw: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None
        confidence_results: List[Dict[str, Any]] = []
        current_section: str = "General"

        for page in self.pages:
            page_num: int = page.get("page_number", 0)
            text: str = page.get("text", "")

            if self._is_table_of_contents(text):
                logger.debug(f"Page {page_num}: Skipping Table of Contents.")
                continue

            if self._is_advertisement(text):
                logger.debug(f"Page {page_num}: Skipping advertisement.")
                continue

            # Continuation page: merge into the last article without creating a new one
            if self._is_continuation_page(text) and current is not None:
                logger.debug(f"Page {page_num}: Merging continuation into '{current['title']}'.")
                cleaned: str = re.sub(
                    r"(continued\s+(from|on)\s+page\s+\d+|\(continued\))",
                    "", text, flags=re.IGNORECASE
                ).strip()
                current["text"] += "\n" + cleaned
                current["end_page"] = page_num
                continue

            for line in text.split("\n"):
                stripped: str = line.strip()
                if not stripped:
                    continue

                # Track section headings without treating them as article starts
                if self._is_section_heading(stripped):
                    current_section = stripped.title()
                    continue

                if self._is_potential_title(stripped):
                    if current and current.get("text", "").strip():
                        raw.append(self._enrich_article(current, confidence_results))

                    score_result: Dict[str, Any] = self._score_title_candidate(stripped)
                    current = {
                        "title": stripped,
                        "section": current_section,
                        "start_page": page_num,
                        "end_page": page_num,
                        "text": "",
                    }
                    confidence_results = [score_result]

                else:
                    if current is None:
                        current = {
                            "title": "Introduction",
                            "section": current_section,
                            "start_page": page_num,
                            "end_page": page_num,
                            "text": "",
                        }
                        confidence_results = [{"score": 0.4, "reason": "Implicit first article."}]

                    current["text"] += stripped + "\n"
                    current["end_page"] = page_num

        # Flush final buffered article
        if current and current.get("text", "").strip():
            raw.append(self._enrich_article(current, confidence_results))

        valid: List[Dict[str, Any]] = [a for a in raw if a.get("word_count", 0) >= 30]
        articles: List[Dict[str, Any]] = self._apply_fallback(valid)

        return {
            "articles": articles,
            "statistics": self._compute_statistics(articles)
        }
