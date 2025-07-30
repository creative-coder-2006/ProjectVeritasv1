import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database Configuration
DATABASE_PATH = "veritas_ai.db"

# API Keys (load from environment variables)
# Note: Reddit API is no longer required - using simplified web scraping
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model Configuration
MISINFORMATION_MODEL = "mrm8488/bert-tiny-fake-news-detection"
LLM_ORIGIN_MODEL = "microsoft/DialoGPT-medium"
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large
VIDEO_ANALYSIS_MODEL = "r3d_18"  # For deepfake detection

# In veritasai/config.py

# ... (other configurations)

FALLBACK_MISINFORMATION_MODEL = "distilbert-base-uncased-finetuned-sst-2-english" # A good fallback

# ... (rest of the file)

# Scraping Configuration
MAX_NEWS_ARTICLES = 50
MAX_REDDIT_POSTS = 100
MAX_REDDIT_COMMENTS = 50
MAX_VIDEOS = 20
MAX_VIDEO_DURATION = 600  # 10 minutes in seconds
MAX_VIDEO_DOWNLOADS = 5  # Maximum videos to download for analysis

# Analysis Configuration
CONFIDENCE_THRESHOLD = 0.7
TRUST_THRESHOLD = 0.6
ENTROPY_THRESHOLD = 0.5
MISINFO_THRESHOLD = 0.7  # Threshold for flagging content as misinformation
LLM_DETECTION_THRESHOLD = 0.7  # Threshold for flagging content as AI-generated

# Subreddit flagging thresholds
SUBREDDIT_FLAG_THRESHOLD = 5  # Number of posts with misinfo_score > 0.7 to flag subreddit
CHANNEL_FLAG_THRESHOLD = 3  # Number of videos with misinfo_score > 0.7 to flag channel

# File Storage
TEMP_DIR = "temp"
AUDIO_DIR = "temp/audio"
VIDEO_DIR = "temp/video"
FRAME_DIR = "temp/frames"

# Streamlit Configuration
PAGE_TITLE = "VERITAS.AI - Truth Detection Platform"
PAGE_ICON = "🔍"

# Subreddits for news and misinformation detection
NEWS_SUBREDDITS = [
    "news", "worldnews", "politics", "technology", 
    "science", "health", "business", "entertainment",
    "conspiracy", "the_donald", "wayofthebern", "sandersforpresident"
]

# Trust scoring weights
TRUST_WEIGHTS = {
    "source_reliability": 0.3,
    "author_credibility": 0.2,
    "cross_check_consistency": 0.25,
    "content_quality": 0.15,
    "temporal_freshness": 0.1
}

# Video analysis configuration
VIDEO_ANALYSIS_CONFIG = {
    "max_frames": 50,
    "frame_interval": 1,  # Extract 1 frame per second
    "face_detection": True,
    "compression_analysis": True,
    "temporal_consistency": True
}

# Reddit analysis configuration
REDDIT_ANALYSIS_CONFIG = {
    "fetch_comments": True,
    "max_comments_per_post": 20,
    "subreddit_metadata": True,
    "engagement_analysis": True,
    "suspicious_patterns": True
}

# OpenAI configuration
OPENAI_CONFIG = {
    "model": "gpt-3.5-turbo",
    "max_tokens": 300,
    "temperature": 0.3,
    "enable_explanations": True
}

# Rate limiting configuration
RATE_LIMITS = {
    "reddit_requests_per_minute": 60,
    "youtube_requests_per_minute": 30,
    "openai_requests_per_minute": 20
} 