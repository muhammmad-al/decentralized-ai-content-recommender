import os
from dotenv import load_dotenv
import praw
import pandas as pd
from datetime import datetime
import time
import logging
from textblob import TextBlob
from transformers import pipeline
import re
from collections import Counter
import numpy as np
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_environment():
    """Load environment variables from .env file"""
    env_path = '.env'
    
    # Debug: Check file existence and location
    current_dir = os.getcwd()
    logger.info(f"Current working directory: {current_dir}")
    logger.info(f"Looking for .env file at: {os.path.join(current_dir, env_path)}")
    
    if not os.path.exists(env_path):
        raise FileNotFoundError(
            "'.env' file not found. Please create one with your Reddit API credentials."
        )
    
    # Load the environment variables
    load_dotenv(env_path)
    
    # Debug: Print loaded variables (be careful not to log actual secrets in production)
    logger.info("Environment variables loaded. Available keys:")
    for key in ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'REDDIT_USER_AGENT']:
        logger.info(f"- {key}: {'Found' if os.getenv(key) else 'Not found'}")

# Define subreddit categories
SUBREDDIT_CATEGORIES = {
    'ai': ["ArtificialIntelligence", "ChatGPT", "MachineLearning"],
    'music': ["hiphopheads", "musicproduction", "FL_Studio"],
    'web3': ["web3", "cryptocurrency", "blockchain"]
}

def initialize_reddit():
    """Initialize and return Reddit instance with credentials from environment variables"""
    try:
        # Debug prints
        client_id = os.getenv('REDDIT_CLIENT_ID')
        client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        user_agent = os.getenv('REDDIT_USER_AGENT')
        
        logger.info(f"Checking credentials:")
        logger.info(f"Client ID exists: {bool(client_id)}")
        logger.info(f"Client Secret exists: {bool(client_secret)}")
        logger.info(f"User Agent exists: {bool(user_agent)}")
        
        if not all([client_id, client_secret, user_agent]):
            raise ValueError("Missing one or more required environment variables")
            
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        return reddit
    except Exception as e:
        logger.error(f"Failed to initialize Reddit instance: {str(e)}")
        raise

def extract_hashtags(text):
    """Extract hashtags from text."""
    if not isinstance(text, str):
        return []
    hashtag_pattern = r'#\w+'
    return re.findall(hashtag_pattern, text)

def clean_text(text):
    """Clean and preprocess text."""
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def get_user_metadata(author):
    """Collect user metadata."""
    if not author or author == '[deleted]':
        return {
            'account_age': None,
            'karma': None,
            'is_verified': None
        }

    try:
        return {
            'account_age': (datetime.utcnow() - datetime.fromtimestamp(author.created_utc)).days,
            'karma': author.link_karma + author.comment_karma,
            'is_verified': author.has_verified_email
        }
    except Exception as e:
        logger.warning(f"Error collecting user metadata: {str(e)}")
        return {
            'account_age': None,
            'karma': None,
            'is_verified': None
        }

def perform_sentiment_analysis(text):
    """Perform sentiment analysis using both TextBlob and Transformers."""
    if not isinstance(text, str) or not text.strip():
        return {
            'textblob_sentiment': 0.0,
            'transformer_sentiment': 'NEUTRAL',
            'transformer_score': 0.5
        }

    try:
        # TextBlob sentiment analysis
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment.polarity

        # Hugging Face sentiment analysis
        classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        transformer_result = classifier(text[:512])[0]  # Limit text length due to model constraints

        return {
            'textblob_sentiment': textblob_sentiment,
            'transformer_sentiment': transformer_result['label'],
            'transformer_score': transformer_result['score']
        }
    except Exception as e:
        logger.warning(f"Error in sentiment analysis: {str(e)}")
        return {
            'textblob_sentiment': 0.0,
            'transformer_sentiment': 'NEUTRAL',
            'transformer_score': 0.5
        }

def collect_posts(reddit, subreddit_name, category, post_limit=100):
    """Collect posts with enhanced data collection and sentiment analysis."""
    posts_data = []
    try:
        subreddit = reddit.subreddit(subreddit_name)
        logger.info(f"Collecting posts from r/{subreddit_name} (Category: {category})")

        for post in subreddit.new(limit=post_limit):
            time.sleep(0.5)  # Respect rate limits

            try:
                # Clean and prepare text
                post_text = post.selftext if hasattr(post, 'selftext') else ''
                cleaned_text = clean_text(post_text)

                # Extract hashtags
                hashtags = extract_hashtags(post_text)

                # Get user metadata
                user_metadata = get_user_metadata(post.author)

                # Perform sentiment analysis
                sentiment_results = perform_sentiment_analysis(cleaned_text)

                # Collect all data
                post_data = {
                    'title': post.title,
                    'cleaned_text': cleaned_text,
                    'original_text': post_text,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'upvote_ratio': post.upvote_ratio,
                    'timestamp': datetime.fromtimestamp(post.created_utc),
                    'author': str(post.author) if post.author else '[deleted]',
                    'hashtags': hashtags,
                    'hashtag_count': len(hashtags),
                    'text_length': len(cleaned_text),
                    'account_age': user_metadata['account_age'],
                    'author_karma': user_metadata['karma'],
                    'author_verified': user_metadata['is_verified'],
                    'textblob_sentiment': sentiment_results['textblob_sentiment'],
                    'transformer_sentiment': sentiment_results['transformer_sentiment'],
                    'transformer_score': sentiment_results['transformer_score'],
                    'category': category,
                    'subreddit': subreddit_name
                }

                posts_data.append(post_data)
                logger.info(f"Collected post from r/{subreddit_name}: {post.title[:50]}...")

            except Exception as e:
                logger.warning(f"Error processing post: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error accessing subreddit {subreddit_name}: {str(e)}")
        return posts_data  # Return any collected data instead of raising

    return posts_data

def save_to_csv(posts_data, category):
    """Save collected posts to CSV file with enhanced error handling."""
    try:
        if not posts_data:
            logger.warning(f"No data to save for category: {category}")
            return

        df = pd.DataFrame(posts_data)

        # Convert timestamp to datetime if not already
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort by timestamp
        df = df.sort_values('timestamp', ascending=False)

        # Create output directory if it doesn't exist
        os.makedirs('data/raw', exist_ok=True)

        # Save to CSV
        filename = f"data/raw/reddit_analysis_{category}.csv"
        df.to_csv(filename, index=False)
        logger.info(f"Successfully saved {len(posts_data)} posts to {filename}")

        # Generate and log basic statistics
        logger.info(f"Dataset Statistics for {category}:")
        logger.info(f"Total posts: {len(df)}")
        logger.info(f"Average sentiment (TextBlob): {df['textblob_sentiment'].mean():.2f}")
        logger.info(f"Most common sentiment (Transformer): {df['transformer_sentiment'].mode()[0]}")
        logger.info(f"Average engagement (score): {df['score'].mean():.2f}")

    except Exception as e:
        logger.error(f"Error saving to CSV: {str(e)}")
        raise

def main():
    try:
        # Load environment variables
        load_environment()

        # Initialize Reddit instance
        reddit = initialize_reddit()

        # Process each category
        for category, subreddits in SUBREDDIT_CATEGORIES.items():
            category_posts = []
            
            for subreddit in subreddits:
                try:
                    posts_data = collect_posts(reddit, subreddit, category)
                    category_posts.extend(posts_data)
                except Exception as e:
                    logger.error(f"Error collecting posts from r/{subreddit}: {str(e)}")
                    continue

            # Save category data
            if category_posts:
                save_to_csv(category_posts, category)
            else:
                logger.error(f"No data collected for category: {category}")

    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()