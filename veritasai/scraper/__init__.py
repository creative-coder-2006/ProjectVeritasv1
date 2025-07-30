# VERITAS.AI Scraper Package
# Contains modules for scraping news, Reddit posts, and YouTube videos

from .news_scraper import fetch_news
from .reddit_scraper import fetch_reddit_posts
from .video_scraper import fetch_videos, download_videos_for_analysis

__all__ = [
    'fetch_news',
    'fetch_reddit_posts', 
    'fetch_videos',
    'download_videos_for_analysis'
] 