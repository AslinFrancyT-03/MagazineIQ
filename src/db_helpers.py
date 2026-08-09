import logging
import sqlite3
import os
import json
from typing import Any, Dict, List, Optional
import streamlit as st

logger = logging.getLogger(__name__)

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'sqlite')
DB_PATH = os.path.join(DB_DIR, 'magazineiq.db')

def get_db_connection():
    """Establishes a connection to the SQLite database with row factory enabled."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes database schema if tables do not exist."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Magazines table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS magazines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                title TEXT,
                status TEXT DEFAULT 'PENDING',
                total_reading_time INTEGER DEFAULT 0,
                avg_complexity REAL DEFAULT 0.0,
                overall_summary TEXT DEFAULT '',
                overall_insights TEXT DEFAULT ''
            )
        """)
        
        # Articles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                magazine_id INTEGER,
                title TEXT,
                content TEXT,
                summary TEXT,
                category TEXT,
                sentiment_score REAL DEFAULT 0.0,
                reading_time INTEGER DEFAULT 0,
                complexity_score REAL DEFAULT 0.0,
                FOREIGN KEY (magazine_id) REFERENCES magazines (id) ON DELETE CASCADE
            )
        """)
        
        # Entities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER,
                name TEXT,
                label TEXT,
                frequency INTEGER DEFAULT 1,
                FOREIGN KEY (article_id) REFERENCES articles (id) ON DELETE CASCADE
            )
        """)
        
        # Keywords table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER,
                phrase TEXT,
                relevance_score REAL,
                FOREIGN KEY (article_id) REFERENCES articles (id) ON DELETE CASCADE
            )
        """)
        
        conn.commit()
    except Exception as e:
        logger.error(f"Error initializing SQLite database: {e}")
        raise
    finally:
        conn.close()

def save_magazine(filename: str, title: str, page_count: int) -> int:
    """Creates a new Magazine record and returns its ID."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO magazines (filename, title, status, overall_summary, overall_insights) VALUES (?, ?, ?, ?, ?)",
            (filename, title, "PENDING", "", json.dumps({"page_count": page_count}))
        )
        conn.commit()
        get_all_magazines.clear()
        return cursor.lastrowid
    except Exception as e:
        logger.error(f"Error saving magazine {filename}: {e}")
        raise
    finally:
        conn.close()

def save_article(magazine_id: int, title: str, content: str, reading_time: int) -> int:
    """Creates a new Article record associated with a magazine and returns its ID."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO articles (magazine_id, title, content, reading_time, summary, category, sentiment_score, complexity_score) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (magazine_id, title, content, reading_time, None, None, 0.0, 0.0)
        )
        conn.commit()
        get_articles.clear()
        return cursor.lastrowid
    except Exception as e:
        logger.error(f"Error saving article '{title}' for magazine {magazine_id}: {e}")
        raise
    finally:
        conn.close()

@st.cache_data(ttl=3600)
def get_magazine(magazine_id: int) -> Optional[Dict[str, Any]]:
    """Loads a magazine record as a dictionary."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, filename, title, status, total_reading_time, avg_complexity, overall_summary, overall_insights FROM magazines WHERE id = ?",
            (magazine_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)
    finally:
        conn.close()

@st.cache_data(ttl=3600)
def get_all_magazines() -> List[Dict[str, Any]]:
    """Lists all magazines currently in the database."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, filename, title, status FROM magazines")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

@st.cache_data(ttl=3600)
def get_articles(magazine_id: int) -> List[Dict[str, Any]]:
    """Loads all articles for a magazine with nested lists of entities and keywords."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, magazine_id, title, content, summary, category, sentiment_score, reading_time, complexity_score FROM articles WHERE magazine_id = ?", (magazine_id,))
        art_rows = cursor.fetchall()
        
        result = []
        for row in art_rows:
            art_id = row['id']
            
            # Load entities
            cursor.execute("SELECT name, label, frequency FROM entities WHERE article_id = ?", (art_id,))
            ent_rows = cursor.fetchall()
            entities = [{"name": e["name"], "label": e["label"], "frequency": e["frequency"]} for e in ent_rows]
            
            # Load keywords
            cursor.execute("SELECT phrase, relevance_score FROM keywords WHERE article_id = ?", (art_id,))
            kw_rows = cursor.fetchall()
            keywords = [{"phrase": k["phrase"], "score": k["relevance_score"]} for k in kw_rows]
            
            art_dict = dict(row)
            art_dict["entities"] = entities
            art_dict["keywords"] = keywords
            result.append(art_dict)
            
        return result
    finally:
        conn.close()

def update_magazine_status(magazine_id: int, status: str):
    """Updates a magazine's status."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE magazines SET status = ? WHERE id = ?", (status, magazine_id))
        conn.commit()
        get_magazine.clear()
        get_all_magazines.clear()
    except Exception as e:
        logger.error(f"Error updating status for magazine {magazine_id}: {e}")
    finally:
        conn.close()

def save_article_summary(article_id: int, summary: str):
    """Saves the generated summary string (or JSON representation) for an article."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE articles SET summary = ? WHERE id = ?", (summary, article_id))
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving summary for article {article_id}: {e}")
    finally:
        conn.close()

def save_article_topic_sentiment(article_id: int, category: str, sentiment_score: float):
    """Saves the zero-shot category and sentiment score for an article."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE articles SET category = ?, sentiment_score = ? WHERE id = ?", (category, sentiment_score, article_id))
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving topic/sentiment for article {article_id}: {e}")
    finally:
        conn.close()

def save_article_keywords(article_id: int, keywords: List[Dict[str, Any]]):
    """Saves keywords for an article, replacing any existing ones."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM keywords WHERE article_id = ?", (article_id,))
        for kw in keywords:
            cursor.execute(
                "INSERT INTO keywords (article_id, phrase, relevance_score) VALUES (?, ?, ?)",
                (article_id, kw["phrase"], kw["score"])
            )
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving keywords for article {article_id}: {e}")
    finally:
        conn.close()

def save_article_entities(article_id: int, entities: List[Dict[str, Any]]):
    """Saves named entities for an article, replacing any existing ones."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM entities WHERE article_id = ?", (article_id,))
        for ent in entities:
            cursor.execute(
                "INSERT INTO entities (article_id, name, label, frequency) VALUES (?, ?, ?, ?)",
                (article_id, ent["name"], ent["label"], ent["frequency"])
            )
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving entities for article {article_id}: {e}")
    finally:
        conn.close()

def update_magazine_stats(magazine_id: int, total_reading_time: int, avg_complexity: float):
    """Updates computed magazine metrics."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE magazines SET total_reading_time = ?, avg_complexity = ? WHERE id = ?", (total_reading_time, avg_complexity, magazine_id))
        conn.commit()
        get_magazine.clear()
        get_all_magazines.clear()
    except Exception as e:
        logger.error(f"Error updating stats for magazine {magazine_id}: {e}")
    finally:
        conn.close()

def save_magazine_meta_summary(magazine_id: int, summary: str, insights: str):
    """Saves the overall magazine meta-summary and insights."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE magazines SET overall_summary = ?, overall_insights = ? WHERE id = ?", (summary, insights, magazine_id))
        conn.commit()
        get_magazine.clear()
    except Exception as e:
        logger.error(f"Error saving magazine meta summary for {magazine_id}: {e}")
    finally:
        conn.close()

def delete_magazine(magazine_id: int):
    """Deletes a magazine and all of its associated articles, entities, and keywords via foreign key cascades."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM magazines WHERE id = ?", (magazine_id,))
        conn.commit()
        get_all_magazines.clear()
        get_magazine.clear()
        get_articles.clear()
    except Exception as e:
        logger.error(f"Error deleting magazine {magazine_id}: {e}")
        raise
    finally:
        conn.close()
