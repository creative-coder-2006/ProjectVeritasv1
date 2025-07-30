import sqlite3
import hashlib
import json
import threading
from datetime import datetime
from config import DATABASE_PATH
import time

# Global connection pool
_connection_pool = {}
_pool_lock = threading.Lock()

def get_db_connection():
    """Get a database connection with proper timeout and error handling."""
    thread_id = threading.get_ident()
    
    with _pool_lock:
        if thread_id in _connection_pool:
            try:
                # Test if connection is still valid
                _connection_pool[thread_id].execute("SELECT 1")
                return _connection_pool[thread_id]
            except (sqlite3.OperationalError, sqlite3.DatabaseError):
                # Connection is invalid, remove it
                del _connection_pool[thread_id]
        
        # Create new connection with timeout
        conn = sqlite3.connect(
            DATABASE_PATH,
            timeout=30.0,  # 30 second timeout
            check_same_thread=False,
            isolation_level=None  # Enable autocommit mode
        )
        
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        
        _connection_pool[thread_id] = conn
        return conn

def close_db_connection():
    """Close the database connection for the current thread."""
    thread_id = threading.get_ident()
    
    with _pool_lock:
        if thread_id in _connection_pool:
            try:
                _connection_pool[thread_id].close()
            except:
                pass
            del _connection_pool[thread_id]

def execute_with_retry(func, max_retries=3, delay=1.0):
    """Execute a database function with retry logic for handling locks."""
    for attempt in range(max_retries):
        try:
            return func()
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                print(f"Database locked, retrying in {delay} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                continue
            else:
                raise e
        except Exception as e:
            print(f"Database error: {e}")
            raise e

def init_database():
    """Initialize the database with all required tables."""
    def _init_db():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # News articles table (enhanced)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                content TEXT,
                source TEXT,
                published_date TIMESTAMP,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                snippet TEXT,
                authors TEXT,
                keywords TEXT,
                summary TEXT,
                is_synthetic BOOLEAN DEFAULT FALSE,
                synthetic_source TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # News sources table (enhanced)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT UNIQUE NOT NULL,
                total_articles INTEGER DEFAULT 0,
                avg_misinfo_score REAL DEFAULT 0.0,
                flagged_as_misinfo BOOLEAN DEFAULT FALSE,
                flag_reason TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                credibility_score REAL DEFAULT 0.0,
                trust_score REAL DEFAULT 0.0,
                flag_count INTEGER DEFAULT 0,
                last_flag_date TIMESTAMP
            )
        ''')
        
        # Misinformation analysis table (enhanced)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS misinfo_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER NOT NULL,
                content_type TEXT DEFAULT 'news',  -- 'news', 'reddit', 'video'
                misinformation_score REAL NOT NULL,
                confidence_score REAL NOT NULL,
                llm_origin_probability REAL,
                trust_score REAL,
                credibility_score REAL,
                analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_version TEXT,
                feature_importance TEXT,  -- JSON string
                openai_explanation TEXT,
                risk_level TEXT,  -- 'HIGH', 'MODERATE', 'LOW'
                FOREIGN KEY (article_id) REFERENCES news_articles (id)
            )
        ''')
        
        # Reddit posts table (enhanced)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reddit_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                content TEXT,
                author TEXT,
                subreddit TEXT,
                score INTEGER,
                created_utc TIMESTAMP,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                is_synthetic BOOLEAN DEFAULT FALSE,
                synthetic_source TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Reddit comments table (NEW)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reddit_comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                comment_id TEXT UNIQUE NOT NULL,
                post_id TEXT NOT NULL,
                content TEXT,
                author TEXT,
                subreddit TEXT,
                score INTEGER,
                created_utc TIMESTAMP,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                FOREIGN KEY (post_id) REFERENCES reddit_posts (post_id),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Subreddit flags table (NEW)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS subreddit_flags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subreddit_name TEXT UNIQUE NOT NULL,
                total_posts INTEGER DEFAULT 0,
                avg_misinfo_score REAL DEFAULT 0.0,
                flagged_as_misinfo BOOLEAN DEFAULT FALSE,
                flag_reason TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                flag_count INTEGER DEFAULT 0,
                last_flag_date TIMESTAMP
            )
        ''')
        
        # YouTube videos table (enhanced)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS youtube_videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                channel TEXT,
                duration INTEGER,
                view_count INTEGER,
                published_at TIMESTAMP,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                transcript TEXT,
                key_frames_path TEXT,
                is_synthetic BOOLEAN DEFAULT FALSE,
                synthetic_source TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Video analysis table (NEW)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                transcript_misinfo_score REAL,
                transcript_confidence REAL,
                transcript_llm_prob REAL,
                deepfake_score REAL,
                deepfake_confidence REAL,
                trust_score REAL,
                credibility_score REAL,
                analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                openai_explanation TEXT,
                risk_level TEXT,
                FOREIGN KEY (video_id) REFERENCES youtube_videos (id)
            )
        ''')
        
        # Channel flags table (NEW)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS channel_flags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_name TEXT UNIQUE NOT NULL,
                total_videos INTEGER DEFAULT 0,
                avg_misinfo_score REAL DEFAULT 0.0,
                avg_deepfake_score REAL DEFAULT 0.0,
                flagged_as_misinfo BOOLEAN DEFAULT FALSE,
                flag_reason TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                flag_count INTEGER DEFAULT 0,
                last_flag_date TIMESTAMP
            )
        ''')
        
        # Coordinated influence operations table (NEW)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS coordinated_operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_id TEXT UNIQUE NOT NULL,
                topic TEXT NOT NULL,
                affected_sources TEXT,  -- JSON array of source names
                affected_platforms TEXT,  -- JSON array of platforms
                total_flagged_content INTEGER DEFAULT 0,
                avg_misinfo_score REAL DEFAULT 0.0,
                detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                severity_level TEXT,  -- 'HIGH', 'MODERATE', 'LOW'
                description TEXT,
                user_id INTEGER,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Synthetic misinformation samples table (NEW)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS synthetic_misinfo (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_type TEXT NOT NULL,  -- 'news', 'reddit', 'video'
                title TEXT NOT NULL,
                content TEXT,
                source TEXT,
                generated_by TEXT,  -- 'llama2', 'mistral', 'gpt'
                generation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                topic TEXT,
                user_id INTEGER,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Analysis results table (enhanced)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id INTEGER NOT NULL,
                content_type TEXT NOT NULL,  -- 'news', 'reddit', 'video'
                misinformation_score REAL,
                confidence_score REAL,
                trust_score REAL,
                credibility_score REAL,
                entropy_score REAL,
                llm_origin_probability REAL,
                deepfake_probability REAL,
                analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                model_version TEXT,
                openai_explanation TEXT,
                risk_level TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # XAI explanations table (enhanced)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS xai_explanations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL,
                explanation_type TEXT NOT NULL,  -- 'misinformation', 'trust', 'credibility'
                explanation_text TEXT,
                feature_importance TEXT,  -- JSON string of feature importance
                ai_explanation TEXT,  -- OpenAI explanation for suspicious articles
                risk_assessment TEXT,
                recommendations TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (analysis_id) REFERENCES analysis_results (id)
            )
        ''')
        
        # History log table (enhanced)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS history_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT NOT NULL,  -- 'scrape', 'analyze', 'flag_source', 'detect_coordination'
                content_type TEXT,  -- 'news', 'reddit', 'video'
                content_id INTEGER,
                user_id INTEGER,
                details TEXT,  -- JSON string with action details
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN DEFAULT TRUE,
                error_message TEXT,
                platform TEXT,  -- 'news', 'reddit', 'youtube'
                source_name TEXT,
                misinfo_score REAL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Platform cross-reference table (NEW - for coordinated operations)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS platform_cross_reference (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                news_sources TEXT,  -- JSON array
                reddit_subreddits TEXT,  -- JSON array
                youtube_channels TEXT,  -- JSON array
                total_flagged INTEGER DEFAULT 0,
                coordination_score REAL DEFAULT 0.0,
                detection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
    
    execute_with_retry(_init_db)

def store_user(username, password_hash):
    """Store a new user in the database."""
    def _store_user():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, password_hash)
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
    
    return execute_with_retry(_store_user)

def verify_user(username, password):
    """Verify user credentials."""
    def _verify_user():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # First get the user's stored password hash
        cursor.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (username,)
        )
        
        user = cursor.fetchone()
        
        if user:
            # Verify the password against the stored hash
            from auth import verify_password
            if verify_password(password, user[2]):  # user[2] is the password_hash
                return (user[0], user[1])  # Return (id, username)
        
        return None
    
    return execute_with_retry(_verify_user)

def store_news_articles(articles, user_id=None):
    """Store news articles in the database."""
    def _store_articles():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        stored_count = 0
        for article in articles:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO news_articles (url, title, content, source, published_date, user_id, snippet, authors, keywords, summary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article['url'], article['title'], article['content'],
                    article['source'], article['published_date'], user_id,
                    article.get('snippet', ''), 
                    ','.join(article.get('authors', [])) if article.get('authors') else '',
                    ','.join(article.get('keywords', [])) if article.get('keywords') else '',
                    article.get('summary', '')
                ))
                if cursor.rowcount > 0:
                    stored_count += 1
                    
                    # Log the successful storage
                    log_action('store_article', 'news', cursor.lastrowid, user_id, 
                              {'title': article['title'], 'source': article['source']})
                
            except sqlite3.IntegrityError:
                # Article already exists, skip
                continue
        
        conn.commit()
        return stored_count
    
    return execute_with_retry(_store_articles)

def store_reddit_posts(posts, user_id=None):
    """Store Reddit posts in the database."""
    def _store_posts():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        for post in posts:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO reddit_posts (post_id, title, content, author, subreddit, score, created_utc, user_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    post['post_id'], post['title'], post['content'],
                    post['author'], post['subreddit'], post['score'],
                    post['created_utc'], user_id
                ))
            except sqlite3.IntegrityError:
                # Post already exists, skip
                continue
        
        conn.commit()
    
    execute_with_retry(_store_posts)

def store_youtube_videos(videos, user_id=None):
    """Store YouTube videos in the database."""
    def _store_videos():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        for video in videos:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO youtube_videos (video_id, title, description, channel, duration, view_count, published_at, user_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    video['video_id'], video['title'], video['description'],
                    video['channel'], video['duration'], video['view_count'],
                    video['published_at'], user_id
                ))
            except sqlite3.IntegrityError:
                # Video already exists, skip
                continue
        
        conn.commit()
    
    execute_with_retry(_store_videos)

def store_analysis_result(content_id, content_type, analysis_data, user_id=None):
    """Store analysis results in the database."""
    def _store_result():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_results 
            (content_id, content_type, misinformation_score, confidence_score, trust_score, 
             credibility_score, entropy_score, llm_origin_probability, deepfake_probability, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            content_id, content_type, analysis_data.get('misinformation_score'),
            analysis_data.get('confidence_score'), analysis_data.get('trust_score'),
            analysis_data.get('credibility_score'), analysis_data.get('entropy_score'),
            analysis_data.get('llm_origin_probability'), analysis_data.get('deepfake_probability'),
            user_id
        ))
        
        analysis_id = cursor.lastrowid
        conn.commit()
        return analysis_id
    
    return execute_with_retry(_store_result)

def store_xai_explanation(analysis_id, explanation_type, explanation_text, feature_importance=None):
    """Store XAI explanations in the database."""
    def _store_explanation():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO xai_explanations (analysis_id, explanation_type, explanation_text, feature_importance)
            VALUES (?, ?, ?, ?)
        ''', (analysis_id, explanation_type, explanation_text, feature_importance))
        
        conn.commit()
    
    execute_with_retry(_store_explanation)

def store_misinfo_analysis(article_id, analysis_data, user_id=None):
    """Store detailed misinformation analysis results."""
    def _store_analysis():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO misinfo_analysis 
            (article_id, misinformation_score, confidence_score, llm_origin_probability, 
             trust_score, credibility_score, model_version, feature_importance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            article_id, analysis_data.get('misinformation_score'),
            analysis_data.get('confidence_score'), analysis_data.get('llm_origin_probability'),
            analysis_data.get('trust_score'), analysis_data.get('credibility_score'),
            analysis_data.get('model_version', 'v1.0'), analysis_data.get('feature_importance', '{}')
        ))
        
        analysis_id = cursor.lastrowid
        conn.commit()
        
        # Log the analysis
        log_action('analyze_misinfo', 'news', article_id, user_id, analysis_data)
        
        return analysis_id
    
    return execute_with_retry(_store_analysis)

def update_news_source_stats(source_name, misinfo_score):
    """Update news source statistics and check for flagging."""
    def _update_stats():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get current stats
        cursor.execute('''
            SELECT total_articles, avg_misinfo_score, flagged_as_misinfo 
            FROM news_sources WHERE source_name = ?
        ''', (source_name,))
        
        result = cursor.fetchone()
        
        if result:
            total_articles, current_avg, is_flagged = result
            new_total = total_articles + 1
            new_avg = ((current_avg * total_articles) + misinfo_score) / new_total
            
            # Check if source should be flagged (3+ articles with misinfo_score > 0.7)
            cursor.execute('''
                SELECT COUNT(*) FROM misinfo_analysis ma
                JOIN news_articles na ON ma.article_id = na.id
                WHERE na.source = ? AND ma.misinformation_score > 0.7
            ''', (source_name,))
            
            high_misinfo_count = cursor.fetchone()[0]
            should_flag = high_misinfo_count >= 3 and not is_flagged
            
            cursor.execute('''
                UPDATE news_sources 
                SET total_articles = ?, avg_misinfo_score = ?, 
                    flagged_as_misinfo = ?, flag_reason = ?, last_updated = CURRENT_TIMESTAMP
                WHERE source_name = ?
            ''', (
                new_total, new_avg, 
                True if should_flag else is_flagged,
                f"Flagged due to {high_misinfo_count} articles with high misinformation scores" if should_flag else None,
                source_name
            ))
            
        else:
            # Create new source entry
            cursor.execute('''
                INSERT INTO news_sources (source_name, total_articles, avg_misinfo_score)
                VALUES (?, 1, ?)
            ''', (source_name, misinfo_score))
        
        conn.commit()
    
    execute_with_retry(_update_stats)

def log_action(action_type, content_type, content_id, user_id, details):
    """Log an action to the history log."""
    def _log_action():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO history_log (action_type, content_type, content_id, user_id, details)
            VALUES (?, ?, ?, ?, ?)
        ''', (action_type, content_type, content_id, user_id, json.dumps(details)))
        
        conn.commit()
    
    execute_with_retry(_log_action)

def get_flagged_sources():
    """Get all flagged news sources."""
    def _get_sources():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT source_name, total_articles, avg_misinfo_score, flag_reason, last_updated
            FROM news_sources 
            WHERE flagged_as_misinfo = TRUE
            ORDER BY avg_misinfo_score DESC
        ''')
        
        results = cursor.fetchall()
        return results
    
    return execute_with_retry(_get_sources)

def get_news_with_analysis(user_id=None, limit=50):
    """Get news articles with their analysis results."""
    def _get_news():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if user_id:
            cursor.execute('''
                SELECT na.*, ma.misinformation_score, ma.confidence_score, 
                       ma.llm_origin_probability, ma.trust_score, ma.credibility_score,
                       ns.flagged_as_misinfo, ns.flag_reason
                FROM news_articles na
                LEFT JOIN misinfo_analysis ma ON na.id = ma.article_id
                LEFT JOIN news_sources ns ON na.source = ns.source_name
                WHERE na.user_id = ?
                ORDER BY na.scraped_at DESC
                LIMIT ?
            ''', (user_id, limit))
        else:
            cursor.execute('''
                SELECT na.*, ma.misinformation_score, ma.confidence_score, 
                       ma.llm_origin_probability, ma.trust_score, ma.credibility_score,
                       ns.flagged_as_misinfo, ns.flag_reason
                FROM news_articles na
                LEFT JOIN misinfo_analysis ma ON na.id = ma.article_id
                LEFT JOIN news_sources ns ON na.source = ns.source_name
                ORDER BY na.scraped_at DESC
                LIMIT ?
            ''', (limit,))
        
        results = cursor.fetchall()
        return results
    
    return execute_with_retry(_get_news)

def get_analysis_history(user_id=None, action_type=None, limit=100):
    """Get analysis history log."""
    def _get_history():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if user_id and action_type:
            cursor.execute('''
                SELECT * FROM history_log 
                WHERE user_id = ? AND action_type = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (user_id, action_type, limit))
        elif user_id:
            cursor.execute('''
                SELECT * FROM history_log 
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (user_id, limit))
        elif action_type:
            cursor.execute('''
                SELECT * FROM history_log 
                WHERE action_type = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (action_type, limit))
        else:
            cursor.execute('''
                SELECT * FROM history_log 
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
        
        results = cursor.fetchall()
        return results
    
    return execute_with_retry(_get_history)

def get_user_content(user_id, content_type=None):
    """Get content for a specific user."""
    def _get_content():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if content_type == 'news':
            cursor.execute('''
                SELECT * FROM news_articles WHERE user_id = ? ORDER BY scraped_at DESC
            ''', (user_id,))
        elif content_type == 'reddit':
            cursor.execute('''
                SELECT * FROM reddit_posts WHERE user_id = ? ORDER BY scraped_at DESC
            ''', (user_id,))
        elif content_type == 'video':
            cursor.execute('''
                SELECT * FROM youtube_videos WHERE user_id = ? ORDER BY scraped_at DESC
            ''', (user_id,))
        else:
            # Get all content types
            cursor.execute('''
                SELECT 'news' as type, id, title, scraped_at FROM news_articles WHERE user_id = ?
                UNION ALL
                SELECT 'reddit' as type, id, title, scraped_at FROM reddit_posts WHERE user_id = ?
                UNION ALL
                SELECT 'video' as type, id, title, scraped_at FROM youtube_videos WHERE user_id = ?
                ORDER BY scraped_at DESC
            ''', (user_id, user_id, user_id))
        
        results = cursor.fetchall()
        return results
    
    return execute_with_retry(_get_content)

def get_analysis_results(user_id, content_type=None):
    """Get analysis results for a specific user."""
    def _get_results():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if content_type:
            cursor.execute('''
                SELECT ar.*, xai.explanation_text, xai.feature_importance
                FROM analysis_results ar
                LEFT JOIN xai_explanations xai ON ar.id = xai.analysis_id
                WHERE ar.user_id = ? AND ar.content_type = ?
                ORDER BY ar.analysis_timestamp DESC
            ''', (user_id, content_type))
        else:
            cursor.execute('''
                SELECT ar.*, xai.explanation_text, xai.feature_importance
                FROM analysis_results ar
                LEFT JOIN xai_explanations xai ON ar.id = xai.analysis_id
                WHERE ar.user_id = ?
                ORDER BY ar.analysis_timestamp DESC
            ''', (user_id,))
        
        results = cursor.fetchall()
        return results
    
    return execute_with_retry(_get_results)

def store_synthetic_misinfo(content_type, title, content, source, generated_by, topic, user_id):
    """Store synthetic misinformation samples."""
    def _store_synthetic_misinfo():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO synthetic_misinfo (content_type, title, content, source, generated_by, topic, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (content_type, title, content, source, generated_by, topic, user_id))
            
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"Error storing synthetic misinformation: {e}")
            return None
    
    return execute_with_retry(_store_synthetic_misinfo)

def store_reddit_comments(comments, user_id):
    """Store Reddit comments."""
    def _store_comments():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        stored_count = 0
        for comment in comments:
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO reddit_comments 
                    (comment_id, post_id, content, author, subreddit, score, created_utc, user_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    comment.get('comment_id'),
                    comment.get('post_id'),
                    comment.get('content'),
                    comment.get('author'),
                    comment.get('subreddit'),
                    comment.get('score'),
                    comment.get('created_utc'),
                    user_id
                ))
                stored_count += 1
            except Exception as e:
                print(f"Error storing comment: {e}")
                continue
        
        conn.commit()
        return stored_count
    
    execute_with_retry(_store_comments)

def update_subreddit_stats(subreddit_name, misinfo_score):
    """Update subreddit statistics and flag if threshold exceeded."""
    def _update_subreddit_stats():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get current stats
            cursor.execute('''
                SELECT total_posts, avg_misinfo_score, flag_count 
                FROM subreddit_flags WHERE subreddit_name = ?
            ''', (subreddit_name,))
            
            result = cursor.fetchone()
            
            if result:
                total_posts, avg_misinfo, flag_count = result
                total_posts += 1
                new_avg = ((avg_misinfo * (total_posts - 1)) + misinfo_score) / total_posts
                
                # Check if should be flagged
                flagged = False
                flag_reason = None
                
                if misinfo_score > 0.7:
                    flag_count += 1
                    if flag_count >= 5:  # Flag if 5+ posts exceed threshold
                        flagged = True
                        flag_reason = f"Multiple posts with high misinformation scores (avg: {new_avg:.2f})"
                
                # Update stats
                cursor.execute('''
                    UPDATE subreddit_flags 
                    SET total_posts = ?, avg_misinfo_score = ?, flag_count = ?, 
                        flagged_as_misinfo = ?, flag_reason = ?, last_updated = CURRENT_TIMESTAMP,
                        last_flag_date = CASE WHEN ? = 1 THEN CURRENT_TIMESTAMP ELSE last_flag_date END
                    WHERE subreddit_name = ?
                ''', (total_posts, new_avg, flag_count, flagged, flag_reason, flagged, subreddit_name))
                
            else:
                # Create new entry
                flagged = misinfo_score > 0.7
                flag_reason = f"High misinformation score: {misinfo_score:.2f}" if flagged else None
                
                cursor.execute('''
                    INSERT INTO subreddit_flags 
                    (subreddit_name, total_posts, avg_misinfo_score, flag_count, 
                     flagged_as_misinfo, flag_reason, last_flag_date)
                    VALUES (?, 1, ?, ?, ?, ?, ?)
                ''', (subreddit_name, misinfo_score, 1 if flagged else 0, flagged, flag_reason, 
                      datetime.now() if flagged else None))
            
            conn.commit()
            return flagged
            
        except Exception as e:
            print(f"Error updating subreddit stats: {e}")
            return False
    
    execute_with_retry(_update_subreddit_stats)

def store_video_analysis(video_id, transcript_analysis, deepfake_analysis, trust_score, credibility_score, openai_explanation, risk_level):
    """Store video analysis results."""
    def _store_video_analysis():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO video_analysis 
                (video_id, transcript_misinfo_score, transcript_confidence, transcript_llm_prob,
                 deepfake_score, deepfake_confidence, trust_score, credibility_score,
                 openai_explanation, risk_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                video_id,
                transcript_analysis.get('misinformation_score', 0),
                transcript_analysis.get('confidence_score', 0),
                transcript_analysis.get('llm_origin_probability', 0),
                deepfake_analysis.get('deepfake_probability', 0),
                deepfake_analysis.get('confidence', 0),
                trust_score,
                credibility_score,
                openai_explanation,
                risk_level
            ))
            
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"Error storing video analysis: {e}")
            return None
    
    execute_with_retry(_store_video_analysis)

def update_channel_stats(channel_name, misinfo_score, deepfake_score):
    """Update channel statistics and flag if threshold exceeded."""
    def _update_channel_stats():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get current stats
            cursor.execute('''
                SELECT total_videos, avg_misinfo_score, avg_deepfake_score, flag_count 
                FROM channel_flags WHERE channel_name = ?
            ''', (channel_name,))
            
            result = cursor.fetchone()
            
            if result:
                total_videos, avg_misinfo, avg_deepfake, flag_count = result
                total_videos += 1
                new_avg_misinfo = ((avg_misinfo * (total_videos - 1)) + misinfo_score) / total_videos
                new_avg_deepfake = ((avg_deepfake * (total_videos - 1)) + deepfake_score) / total_videos
                
                # Check if should be flagged
                flagged = False
                flag_reason = None
                
                if misinfo_score > 0.7 or deepfake_score > 0.7:
                    flag_count += 1
                    if flag_count >= 3:  # Flag if 3+ videos exceed threshold
                        flagged = True
                        flag_reason = f"Multiple videos with high scores (misinfo: {new_avg_misinfo:.2f}, deepfake: {new_avg_deepfake:.2f})"
                
                # Update stats
                cursor.execute('''
                    UPDATE channel_flags 
                    SET total_videos = ?, avg_misinfo_score = ?, avg_deepfake_score = ?, flag_count = ?, 
                        flagged_as_misinfo = ?, flag_reason = ?, last_updated = CURRENT_TIMESTAMP,
                        last_flag_date = CASE WHEN ? = 1 THEN CURRENT_TIMESTAMP ELSE last_flag_date END
                    WHERE channel_name = ?
                ''', (total_videos, new_avg_misinfo, new_avg_deepfake, flag_count, flagged, flag_reason, flagged, channel_name))
                
            else:
                # Create new entry
                flagged = misinfo_score > 0.7 or deepfake_score > 0.7
                flag_reason = f"High scores (misinfo: {misinfo_score:.2f}, deepfake: {deepfake_score:.2f})" if flagged else None
                
                cursor.execute('''
                    INSERT INTO channel_flags 
                    (channel_name, total_videos, avg_misinfo_score, avg_deepfake_score, flag_count, 
                     flagged_as_misinfo, flag_reason, last_flag_date)
                    VALUES (?, 1, ?, ?, ?, ?, ?, ?)
                ''', (channel_name, misinfo_score, deepfake_score, 1 if flagged else 0, flagged, flag_reason, 
                      datetime.now() if flagged else None))
            
            conn.commit()
            return flagged
            
        except Exception as e:
            print(f"Error updating channel stats: {e}")
            return False
    
    execute_with_retry(_update_channel_stats)

def detect_coordinated_operations(topic, user_id):
    """Detect coordinated influence operations across platforms."""
    def _detect_coordinated_operations():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Get flagged sources from all platforms
            flagged_sources = []
            
            # News sources
            cursor.execute('''
                SELECT source_name, avg_misinfo_score FROM news_sources 
                WHERE flagged_as_misinfo = 1
            ''')
            news_sources = cursor.fetchall()
            for source in news_sources:
                flagged_sources.append({'platform': 'news', 'source': source[0], 'score': source[1]})
            
            # Reddit subreddits
            cursor.execute('''
                SELECT subreddit_name, avg_misinfo_score FROM subreddit_flags 
                WHERE flagged_as_misinfo = 1
            ''')
            reddit_sources = cursor.fetchall()
            for source in reddit_sources:
                flagged_sources.append({'platform': 'reddit', 'source': source[0], 'score': source[1]})
            
            # YouTube channels
            cursor.execute('''
                SELECT channel_name, avg_misinfo_score FROM channel_flags 
                WHERE flagged_as_misinfo = 1
            ''')
            youtube_sources = cursor.fetchall()
            for source in youtube_sources:
                flagged_sources.append({'platform': 'youtube', 'source': source[0], 'score': source[1]})
            
            # Check for coordination (2+ platforms with flagged sources)
            platforms_with_flags = set(source['platform'] for source in flagged_sources)
            
            if len(platforms_with_flags) >= 2:
                # Calculate coordination score
                avg_scores = [source['score'] for source in flagged_sources]
                coordination_score = sum(avg_scores) / len(avg_scores) if avg_scores else 0
                
                # Determine severity
                if coordination_score > 0.8:
                    severity = 'HIGH'
                elif coordination_score > 0.6:
                    severity = 'MODERATE'
                else:
                    severity = 'LOW'
                
                # Store coordinated operation
                operation_id = f"coord_{topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                cursor.execute('''
                    INSERT INTO coordinated_operations 
                    (operation_id, topic, affected_sources, affected_platforms, total_flagged_content,
                     avg_misinfo_score, severity_level, description, user_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    operation_id,
                    topic,
                    json.dumps([source['source'] for source in flagged_sources]),
                    json.dumps(list(platforms_with_flags)),
                    len(flagged_sources),
                    coordination_score,
                    severity,
                    f"Coordinated influence operation detected across {len(platforms_with_flags)} platforms",
                    user_id
                ))
                
                conn.commit()
                return {
                    'detected': True,
                    'operation_id': operation_id,
                    'severity': severity,
                    'platforms': list(platforms_with_flags),
                    'sources': [source['source'] for source in flagged_sources],
                    'coordination_score': coordination_score
                }
            
            return {'detected': False}
            
        except Exception as e:
            print(f"Error detecting coordinated operations: {e}")
            return {'detected': False, 'error': str(e)}
    
    execute_with_retry(_detect_coordinated_operations)

def get_coordinated_operations(user_id, limit=10):
    """Get recent coordinated operations."""
    def _get_coordinated_operations():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT operation_id, topic, affected_sources, affected_platforms, 
                       total_flagged_content, avg_misinfo_score, severity_level, 
                       description, detection_timestamp
                FROM coordinated_operations 
                WHERE user_id = ?
                ORDER BY detection_timestamp DESC
                LIMIT ?
            ''', (user_id, limit))
            
            results = cursor.fetchall()
            return results
        except Exception as e:
            print(f"Error getting coordinated operations: {e}")
            return []
    
    execute_with_retry(_get_coordinated_operations)

def get_flagged_sources_by_platform(platform):
    """Get flagged sources for a specific platform."""
    def _get_flagged_sources():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            if platform == 'news':
                cursor.execute('''
                    SELECT source_name, total_articles, avg_misinfo_score, flag_reason, last_updated
                    FROM news_sources WHERE flagged_as_misinfo = 1
                    ORDER BY avg_misinfo_score DESC
                ''')
            elif platform == 'reddit':
                cursor.execute('''
                    SELECT subreddit_name, total_posts, avg_misinfo_score, flag_reason, last_updated
                    FROM subreddit_flags WHERE flagged_as_misinfo = 1
                    ORDER BY avg_misinfo_score DESC
                ''')
            elif platform == 'youtube':
                cursor.execute('''
                    SELECT channel_name, total_videos, avg_misinfo_score, flag_reason, last_updated
                    FROM channel_flags WHERE flagged_as_misinfo = 1
                    ORDER BY avg_misinfo_score DESC
                ''')
            else:
                return []
            
            return cursor.fetchall()
        except Exception as e:
            print(f"Error getting flagged sources: {e}")
            return []
    
    execute_with_retry(_get_flagged_sources)

def get_synthetic_misinfo_samples(content_type, topic, user_id, limit=5):
    """Get synthetic misinformation samples for comparison."""
    def _get_synthetic_samples():
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT title, content, source, generated_by, generation_timestamp
                FROM synthetic_misinfo 
                WHERE content_type = ? AND topic = ? AND user_id = ?
                ORDER BY generation_timestamp DESC
                LIMIT ?
            ''', (content_type, topic, user_id, limit))
            
            return cursor.fetchall()
        except Exception as e:
            print(f"Error getting synthetic samples: {e}")
            return []
    
    execute_with_retry(_get_synthetic_samples)

def log_coordination_detection(topic, platforms, sources, coordination_score, user_id):
    """Log coordination detection event."""
    log_action(
        'detect_coordination',
        'coordination',
        None,
        user_id,
        {
            'topic': topic,
            'platforms': platforms,
            'sources': sources,
            'coordination_score': coordination_score
        }
    ) 