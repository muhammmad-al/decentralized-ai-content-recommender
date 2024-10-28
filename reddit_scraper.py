import praw
import pandas as pd
from datetime import datetime
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_reddit():
    """Initialize and return Reddit instance with provided credentials."""
    try:
        reddit = praw.Reddit(
            client_id="KA872pVkyZiv-pMWiULNsw",
            client_secret="8D3IzS2P_XZ5Jf5FuGBrGdFZ9vxsfg",
            user_agent="web3_scraper/1.0 (by /u/web3_scraper)"
        )
        return reddit
    except Exception as e:
        logger.error(f"Failed to initialize Reddit instance: {str(e)}")
        raise

def collect_posts(reddit, subreddit_name, post_limit=100):
    """Collect posts from specified subreddit."""
    posts_data = []
    try:
        subreddit = reddit.subreddit(subreddit_name)

        for post in subreddit.new(limit=post_limit):
            # Handle rate limits
            time.sleep(0.5)  # Add delay to respect rate limits
            
            try:
                post_data = {
                    'title': post.title,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'timestamp': datetime.fromtimestamp(post.created_utc),
                    'post_text': post.selftext if hasattr(post, 'selftext') else '',
                    'author': str(post.author) if post.author else '[deleted]'
                }
                posts_data.append(post_data)
                logger.info(f"Collected post: {post.title[:50]}...")

            except Exception as e:
                logger.warning(f"Error processing post: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error accessing subreddit: {str(e)}")
        raise

    return posts_data

def save_to_csv(posts_data, filename):
    """Save collected posts to CSV file."""
    try:
        df = pd.DataFrame(posts_data)
        df.to_csv(filename, index=False)
        logger.info(f"Successfully saved {len(posts_data)} posts to {filename}")
    except Exception as e:
        logger.error(f"Error saving to CSV: {str(e)}")
        raise

def main():
    try:
        # Initialize Reddit instance
        reddit = initialize_reddit()
        
        # Collect posts
        logger.info("Starting to collect posts from r/web3...")
        posts_data = collect_posts(reddit, "web3")
        
        # Save to CSV
        save_to_csv(posts_data, "reddit_posts.csv")
        
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
