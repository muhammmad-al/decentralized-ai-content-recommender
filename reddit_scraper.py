import os
from dotenv import load_dotenv
import praw
import pandas as pd
from datetime import datetime
import asyncio
import aiohttp
import logging
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from functools import partial

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SUBREDDIT_CATEGORIES = {
    'ai': [
        "ChatGPT",  
        "artificial", 
        "MachineLearning"
    ],
    'music': [
        "hiphopheads",  
        "WeAreTheMusicMakers",
        "edmproduction"
    ],
    'web3': [
        "cryptocurrency",  
        "ethereum",
        "CryptoTechnology",  
        "defi",             
        "NFT"               
    ]
}

def initialize_reddit():
    """Initialize Reddit instance"""
    return praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        user_agent=os.getenv('REDDIT_USER_AGENT')
    )

def clean_text(text):
    """Minimal text cleaning"""
    if not isinstance(text, str):
        return ""
    return ' '.join(text.lower().split())

def process_post(post):
    """Process a single post with minimal computation"""
    try:
        return {
            'title': post.title,
            'cleaned_text': clean_text(post.selftext if hasattr(post, 'selftext') else ''),
            'score': post.score,
            'num_comments': post.num_comments,
            'upvote_ratio': post.upvote_ratio,
            'timestamp': datetime.fromtimestamp(post.created_utc),
            'category': post.category,
            'subreddit': post.subreddit.display_name
        }
    except Exception:
        return None

def collect_subreddit_posts(reddit, subreddit_name, category, post_limit=400):
    """Collect posts from a single subreddit"""
    try:
        subreddit = reddit.subreddit(subreddit_name)
        posts = []
        
        # Collect only from 'hot' for speed
        for post in subreddit.hot(limit=post_limit):
            post.category = category  # Add category to post object
            posts.append(post)
            
        return posts
    except Exception as e:
        logger.error(f"Error collecting from {subreddit_name}: {str(e)}")
        return []

def parallel_collect_posts(reddit, category, subreddits):
    """Collect posts in parallel from multiple subreddits"""
    with ThreadPoolExecutor(max_workers=len(subreddits)) as executor:
        # Create partial function with fixed arguments
        collect_func = partial(collect_subreddit_posts, reddit, category=category)
        
        # Map subreddits to collection function
        future_to_subreddit = {
            executor.submit(collect_func, subreddit): subreddit 
            for subreddit in subreddits
        }
        
        all_posts = []
        for future in concurrent.futures.as_completed(future_to_subreddit):
            subreddit = future_to_subreddit[future]
            try:
                posts = future.result()
                all_posts.extend(posts)
                logger.info(f"Collected {len(posts)} posts from r/{subreddit}")
            except Exception as e:
                logger.error(f"Error processing r/{subreddit}: {str(e)}")
                
        return all_posts

def process_posts_parallel(posts):
    """Process posts in parallel"""
    with ThreadPoolExecutor() as executor:
        processed = list(executor.map(process_post, posts))
    return [p for p in processed if p is not None]

def save_to_csv(posts_data, category):
    """Quick save to CSV with minimal processing"""
    if not posts_data:
        return
        
    df = pd.DataFrame(posts_data)
    
    # Basic cleaning
    df = df.dropna(subset=['cleaned_text'])
    df = df.drop_duplicates(subset=['title', 'cleaned_text'])
    
    # Sort by engagement
    df['engagement'] = df['score'] + df['num_comments']
    df = df.sort_values('engagement', ascending=False)
    
    # Keep top posts
    df = df.head(1000)
    
    # Save
    os.makedirs('data/raw', exist_ok=True)
    filename = f"data/raw/reddit_analysis_{category}.csv"
    df.to_csv(filename, index=False)
    
    logger.info(f"Saved {len(df)} posts for {category}")

async def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize Reddit
    reddit = initialize_reddit()
    
    for category, subreddits in SUBREDDIT_CATEGORIES.items():
        logger.info(f"\nCollecting {category} posts...")
        
        # Collect posts in parallel
        raw_posts = parallel_collect_posts(reddit, category, subreddits)
        logger.info(f"Collected {len(raw_posts)} total posts for {category}")
        
        # Process posts in parallel
        processed_posts = process_posts_parallel(raw_posts)
        logger.info(f"Processed {len(processed_posts)} posts for {category}")
        
        # Save results
        save_to_csv(processed_posts, category)

if __name__ == "__main__":
    asyncio.run(main())