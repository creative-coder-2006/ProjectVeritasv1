import requests
import feedparser
from newspaper import Article
from datetime import datetime
import time
from config import MAX_NEWS_ARTICLES

def fetch_news_rss(topic, max_articles=MAX_NEWS_ARTICLES):
    """Fetch news articles from Google News RSS feed."""
    # URL encode the topic to handle spaces and special characters
    import urllib.parse
    encoded_topic = urllib.parse.quote(topic)
    # Google News RSS URL
    rss_url = f"https://news.google.com/rss/search?q={encoded_topic}&hl=en-US&gl=US&ceid=US:en"
    
    try:
        # Parse RSS feed
        feed = feedparser.parse(rss_url)
        articles = []
        
        for entry in feed.entries[:max_articles]:
            try:
                # Extract basic info from RSS
                article_data = {
                    'title': entry.title,
                    'url': entry.link,
                    'source': entry.source.title if hasattr(entry, 'source') else 'Unknown',
                    'published_date': datetime.fromtimestamp(time.mktime(entry.published_parsed)) if hasattr(entry, 'published_parsed') else None,
                    'snippet': entry.summary if hasattr(entry, 'summary') else ''
                }
                
                # Try to extract full content using newspaper3k
                try:
                    article = Article(entry.link)
                    article.download()
                    article.parse()
                    
                    if article.text:
                        article_data['content'] = article.text
                        article_data['authors'] = article.authors
                        article_data['keywords'] = article.keywords
                        article_data['summary'] = article.summary
                    else:
                        article_data['content'] = article_data['snippet']
                        
                except Exception as e:
                    # Fallback to snippet if full content extraction fails
                    article_data['content'] = article_data['snippet']
                
                articles.append(article_data)
                
            except Exception as e:
                print(f"Error processing article: {e}")
                continue
        
        return articles
        
    except Exception as e:
        print(f"Error fetching RSS feed: {e}")
        return []

def fetch_news_from_multiple_sources(topic, max_articles=MAX_NEWS_ARTICLES):
    """Fetch news from multiple authentic sources for better coverage."""
    articles = []
    
    # Google News (primary source)
    print(f"Fetching from Google News...")
    google_articles = fetch_news_rss(topic, max_articles // 2)
    articles.extend(google_articles)
    
    # BBC News RSS
    try:
        print(f"Fetching from BBC News...")
        bbc_url = f"https://feeds.bbci.co.uk/news/rss.xml?edition=us"
        bbc_feed = feedparser.parse(bbc_url)
        bbc_articles = []
        
        for entry in bbc_feed.entries[:max_articles // 4]:
            if topic.lower() in entry.title.lower() or topic.lower() in entry.summary.lower():
                try:
                    article_data = {
                        'title': entry.title,
                        'url': entry.link,
                        'source': 'BBC News',
                        'published_date': datetime.fromtimestamp(time.mktime(entry.published_parsed)) if hasattr(entry, 'published_parsed') else None,
                        'snippet': entry.summary,
                        'content': entry.summary  # Start with summary, will be enhanced later
                    }
                    
                    # Try to get full content
                    try:
                        article = Article(entry.link)
                        article.download()
                        article.parse()
                        if article.text:
                            article_data['content'] = article.text
                            article_data['authors'] = article.authors
                            article_data['keywords'] = article.keywords
                            article_data['summary'] = article.summary
                    except:
                        pass
                    
                    bbc_articles.append(article_data)
                except Exception as e:
                    continue
        
        articles.extend(bbc_articles)
        print(f"Found {len(bbc_articles)} BBC articles")
        
    except Exception as e:
        print(f"Error fetching BBC News: {e}")
    
    # Reuters RSS
    try:
        print(f"Fetching from Reuters...")
        reuters_url = f"https://feeds.reuters.com/Reuters/worldNews"
        reuters_feed = feedparser.parse(reuters_url)
        reuters_articles = []
        
        for entry in reuters_feed.entries[:max_articles // 4]:
            if topic.lower() in entry.title.lower() or topic.lower() in entry.summary.lower():
                try:
                    article_data = {
                        'title': entry.title,
                        'url': entry.link,
                        'source': 'Reuters',
                        'published_date': datetime.fromtimestamp(time.mktime(entry.published_parsed)) if hasattr(entry, 'published_parsed') else None,
                        'snippet': entry.summary,
                        'content': entry.summary
                    }
                    
                    # Try to get full content
                    try:
                        article = Article(entry.link)
                        article.download()
                        article.parse()
                        if article.text:
                            article_data['content'] = article.text
                            article_data['authors'] = article.authors
                            article_data['keywords'] = article.keywords
                            article_data['summary'] = article.summary
                    except:
                        pass
                    
                    reuters_articles.append(article_data)
                except Exception as e:
                    continue
        
        articles.extend(reuters_articles)
        print(f"Found {len(reuters_articles)} Reuters articles")
        
    except Exception as e:
        print(f"Error fetching Reuters: {e}")
    
    # Remove duplicates based on URL
    seen_urls = set()
    unique_articles = []
    
    for article in articles:
        if article['url'] not in seen_urls:
            seen_urls.add(article['url'])
            unique_articles.append(article)
    
    return unique_articles[:max_articles]

def extract_article_metadata(article_url):
    """Extract detailed metadata from a specific article URL."""
    try:
        article = Article(article_url)
        article.download()
        article.parse()
        
        return {
            'title': article.title,
            'url': article_url,
            'content': article.text,
            'source': article.source_url,
            'published_date': article.publish_date,
            'authors': article.authors,
            'keywords': article.keywords,
            'summary': article.summary
        }
        
    except Exception as e:
        print(f"Error extracting metadata from {article_url}: {e}")
        return None

def validate_article_content(article):
    """Validate article content for quality and completeness."""
    if not article.get('title') or not article.get('content'):
        return False
    
    # Check minimum content length
    if len(article['content']) < 100:
        return False
    
    # Check for common spam indicators
    spam_indicators = ['click here', 'buy now', 'limited time', 'act now']
    content_lower = article['content'].lower()
    
    spam_score = sum(1 for indicator in spam_indicators if indicator in content_lower)
    if spam_score > 2:
        return False
    
    return True

def fetch_news(topic, max_articles=MAX_NEWS_ARTICLES):
    """Main function to fetch news articles for a given topic."""
    print(f"Fetching news articles for topic: {topic}")
    
    # Fetch articles from multiple sources
    articles = fetch_news_from_multiple_sources(topic, max_articles)
    
    # Validate and filter articles
    valid_articles = []
    for article in articles:
        if validate_article_content(article):
            valid_articles.append(article)
    
    print(f"Found {len(valid_articles)} valid articles out of {len(articles)} total")
    return valid_articles 