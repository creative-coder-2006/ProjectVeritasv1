# In scraper/reddit_scraper.py

import pandas as pd
import time

def fetch_reddit_posts(topic, limit=5):
    """
    Fetches posts from Reddit related to a specific topic using simulated data.
    This is a simplified version for demonstration and fallback purposes.
    """
    print(f"Fetching Reddit posts for topic: {topic} (using simulated data)")
    
    # Simulate a network delay
    time.sleep(2)

    # Create a list of simulated Reddit posts
    posts_data = [
        {
            "title": f"This is a simulated Reddit post about {topic}",
            "author": "user123",
            "created_utc": pd.to_datetime("2023-10-27T10:00:00Z").timestamp(),
            "score": 150,
            "num_comments": 45,
            "selftext": f"Detailed discussion about {topic}. This content is generated for demonstration purposes as the real scraper failed to load.",
            "url": "https://www.reddit.com/r/example/comments/1/simulated_post_1"
        },
        {
            "title": f"Another perspective on {topic}",
            "author": "commenter_pro",
            "created_utc": pd.to_datetime("2023-10-27T12:30:00Z").timestamp(),
            "score": 88,
            "num_comments": 22,
            "selftext": f"Here is another take on {topic}. This is part of the simplified analysis mode.",
            "url": "https://www.reddit.com/r/example/comments/2/simulated_post_2"
        }
    ]

    # Limit the number of posts
    posts_data = posts_data[:limit]

    # Convert to a DataFrame
    df = pd.DataFrame(posts_data)
    
    # Convert timestamp to datetime
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
    
    print(f"Found {len(df)} Reddit posts.")
    return df