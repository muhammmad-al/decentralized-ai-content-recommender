import os
from dotenv import load_dotenv
import praw
import pandas as pd
from datetime import datetime
import asyncio
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
        "MachineLearning",
        "AIdev",
        "deeplearning",
        "OpenAI",
        "reinforcementlearning",
        "GPT3",
        "ArtificialInteligence",
        "MLQuestions"
    ],
    'music': [
        "hiphopheads",
        "WeAreTheMusicMakers",
        "edmproduction",
        "Music",
        "musicproduction",
        "IndieMusicFeedback",
        "musictheory",
        "makinghiphop",
        "FL_Studio",
        "ableton"
    ],
    'web3': [
        "cryptocurrency",
        "ethereum",
        "CryptoTechnology",
        "defi",
        "NFT",
        "web3",
        "solana",
        "bitcoindev",
        "ethdev",
        "BlockchainStartups"
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
            'subreddit': post.subreddit.display_name,
            'author': str(post.author) if post.author else '[deleted]',
            'is_original_content': post.is_original_content if hasattr(post, 'is_original_content') else False
        }
    except Exception:
        return None

def collect_subreddit_posts(reddit, subreddit_name, category, post_limit=1000):
    """Collect posts from a single subreddit using multiple sorting methods"""
    try:
        subreddit = reddit.subreddit(subreddit_name)
        posts = set()  # Use set to avoid duplicates
        
        # Collect from multiple sorting methods to get more diverse posts
        sorting_methods = {
            'hot': 400,
            'top': 300,
            'new': 200,
            'rising': 100
        }
        
        for sort_method, limit in sorting_methods.items():
            try:
                if sort_method == 'top':
                    # Get posts from different time periods
                    for time_filter in ['month', 'year', 'all']:
                        method = getattr(subreddit, sort_method)
                        for post in method(limit=limit, time_filter=time_filter):
                            post.category = category
                            posts.add(post)
                else:
                    method = getattr(subreddit, sort_method)
                    for post in method(limit=limit):
                        post.category = category
                        posts.add(post)
            except Exception as e:
                logger.warning(f"Error collecting {sort_method} posts from {subreddit_name}: {str(e)}")
                continue
                
        return list(posts)
    except Exception as e:
        logger.error(f"Error collecting from {subreddit_name}: {str(e)}")
        return []

def parallel_collect_posts(reddit, category, subreddits):
    """Collect posts in parallel from multiple subreddits"""
    with ThreadPoolExecutor(max_workers=min(len(subreddits), 10)) as executor:
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
    """Save to CSV with enhanced processing"""
    if not posts_data:
        return
        
    df = pd.DataFrame(posts_data)
    
    # Enhanced cleaning
    df = df.dropna(subset=['cleaned_text'])
    df = df.drop_duplicates(subset=['title', 'cleaned_text'])
    
    # Calculate engagement score
    df['engagement'] = (
        df['score'] * 0.6 +  # Weight score more heavily
        df['num_comments'] * 0.4 +  # Comments contribute less
        (df['upvote_ratio'] * 10)  # Small boost for highly upvoted content
    )
    
    # Sort by engagement and keep top 5000 posts
    df = df.sort_values('engagement', ascending=False)
    df = df.head(5000)
    
    # Save
    os.makedirs('data/raw', exist_ok=True)
    filename = f"data/raw/reddit_analysis_{category}_{len(df)}_posts.csv"
    df.to_csv(filename, index=False)
    
    logger.info(f"Saved {len(df)} posts for {category}")
    return len(df)

async def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize Reddit
    reddit = initialize_reddit()
    
    total_posts = 0
    for category, subreddits in SUBREDDIT_CATEGORIES.items():
        logger.info(f"\nCollecting {category} posts...")
        
        # Collect posts in parallel
        raw_posts = parallel_collect_posts(reddit, category, subreddits)
        logger.info(f"Collected {len(raw_posts)} total posts for {category}")
        
        # Process posts in parallel
        processed_posts = process_posts_parallel(raw_posts)
        logger.info(f"Processed {len(processed_posts)} posts for {category}")
        
        # Save results
        posts_saved = save_to_csv(processed_posts, category)
        total_posts += posts_saved
        
    logger.info(f"\nScript completed. Total posts collected: {total_posts}")

if __name__ == "__main__":
    asyncio.run(main())