import math
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)

class ScoringEngine:
    """
    Computes advanced offline metrics like Magazine DNA and Intelligence Score.
    """
    
    DNA_CATEGORIES = [
        "Technology", "Business", "Politics", "Science", 
        "Entertainment", "Health", "Sports", "Finance"
    ]

    def compute_magazine_dna(self, articles: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Maps article topics to the 8 DNA categories and returns a percentage breakdown.
        """
        if not articles:
            return {cat: 0.0 for cat in self.DNA_CATEGORIES}
            
        counts = {cat: 0 for cat in self.DNA_CATEGORIES}
        
        # Simple mapping of broad topics to DNA categories
        topic_mapping = {
            "business": "Business",
            "technology": "Technology",
            "science": "Science",
            "health": "Health",
            "politics": "Politics",
            "entertainment": "Entertainment",
            "sports": "Sports",
            "finance": "Finance",
            "economy": "Finance",
            "culture": "Entertainment",
            "art": "Entertainment",
            "environment": "Science",
            "education": "Science"
        }
        
        total_matched = 0
        for art in articles:
            topic = art.get("category")
            if topic:
                mapped = topic_mapping.get(topic.lower())
                if mapped:
                    counts[mapped] += 1
                    total_matched += 1
                
        if total_matched == 0:
            return {cat: 12.5 for cat in self.DNA_CATEGORIES} # Equal distribution if no match
            
        return {cat: (count / total_matched) * 100 for cat, count in counts.items()}

    def compute_intelligence_score(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Computes the custom Intelligence Score algorithm.
        Returns the overall score (0-100) and the 8 component scores.
        """
        if not articles:
            return {"overall_score": 0, "components": {}}
            
        n = len(articles)
        
        # 1. Readability (Flesch proxy)
        # Assuming avg English word is 5 chars, avg sentence is 15 words
        readability_scores = []
        for a in articles:
            text = a.get("content", "")
            words = len(text.split())
            sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
            avg_words_per_sentence = words / sentences
            # Proxy: shorter sentences = higher readability score (capped at 100)
            score = max(0, min(100, 100 - (avg_words_per_sentence - 10) * 2))
            readability_scores.append(score)
        readability = sum(readability_scores) / n if n else 0

        # 2. Vocabulary Richness (Type-Token Ratio)
        ttr_scores = []
        for a in articles:
            tokens = a.get("content", "").lower().split()
            if tokens:
                unique_tokens = len(set(tokens))
                ttr = unique_tokens / len(tokens)
                # TTR usually falls between 0.3 (rich) and 0.1 (repetitive) for long texts
                # Map to 0-100
                score = min(100, max(0, (ttr - 0.1) * 500))
                ttr_scores.append(score)
        vocab_richness = sum(ttr_scores) / n if n else 0

        # 3. Topic Diversity
        topics = [a.get("category") for a in articles if a.get("category")]
        unique_topics = len(set(topics))
        topic_diversity = min(100, (unique_topics / 8) * 100) # Capped against 8 broad categories

        # 4. Entity Richness
        entity_scores = []
        for a in articles:
            words = len(a.get("content", "").split())
            entities = len(a.get("entities", []))
            if words > 0:
                density = (entities / words) * 1000 # Entities per 1000 words
                score = min(100, (density / 50) * 100) # Assuming 50 per 1000 is excellent
                entity_scores.append(score)
        entity_richness = sum(entity_scores) / n if n else 0

        # 5. Information Density (Compression ratio proxy)
        density_scores = []
        for a in articles:
            summary_str = a.get("summary")
            if summary_str:
                try:
                    import json
                    summ_data = json.loads(summary_str)
                    if isinstance(summ_data, dict):
                        stats = summ_data.get("summary_statistics", {})
                        ratio = stats.get("compression_ratio", 0)
                        if ratio > 0:
                            score = min(100, max(0, (1 - ratio) * 120))
                            density_scores.append(score)
                except Exception:
                    pass
        info_density = sum(density_scores) / n if n else 0

        # 6. Keyword Coverage
        kws_scores = []
        for a in articles:
            kws = len(a.get("keywords", []))
            score = min(100, (kws / 15) * 100) # 15 good keywords is max score
            kws_scores.append(score)
        keyword_coverage = sum(kws_scores) / n if n else 0

        # 7. Content Quality (Model Confidence)
        quality_scores = []
        for a in articles:
            topic = a.get("category")
            if topic:
                # Since we only store the label string in DB and not the raw model confidence,
                # we assign a sensible default quality score of 85% for successfully classified text.
                quality_scores.append(85.0)
            else:
                quality_scores.append(0.0)
        content_quality = sum(quality_scores) / n if n else 0

        # 8. Writing Consistency (Variance in word counts)
        word_counts = [len(a.get("content", "").split()) for a in articles]
        if n > 1:
            mean_wc = sum(word_counts) / n
            variance = sum((x - mean_wc) ** 2 for x in word_counts) / n
            std_dev = math.sqrt(variance)
            # Lower standard deviation relative to mean = higher consistency
            cv = std_dev / mean_wc if mean_wc else 1
            consistency = max(0, 100 - (cv * 100))
        else:
            consistency = 100.0

        components = {
            "Readability": round(readability, 1),
            "Vocabulary Richness": round(vocab_richness, 1),
            "Topic Diversity": round(topic_diversity, 1),
            "Entity Richness": round(entity_richness, 1),
            "Information Density": round(info_density, 1),
            "Keyword Coverage": round(keyword_coverage, 1),
            "Content Quality": round(content_quality, 1),
            "Writing Consistency": round(consistency, 1)
        }
        
        overall = round(sum(components.values()) / 8, 1)
        
        return {
            "overall_score": overall,
            "components": components
        }
