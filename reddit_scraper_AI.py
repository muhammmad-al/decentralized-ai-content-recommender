import praw
import pandas as pd
from datetime import datetime
import time
import logging
from textblob import TextBlob  # For basic sentiment analysis
from transformers import pipeline  # For more advanced sentiment analysis
import re
from collections import Counter
import numpy as np
import emoji
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

def extract_hashtags(text):
    """Extract hashtags from text."""
    hashtag_pattern = r'#\w+'
    return re.findall(hashtag_pattern, text)

def clean_text(text):
    """Clean and preprocess text."""
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
            'textblob_sentiment': None,
            'transformer_sentiment': None,
            'transformer_score': None
        }

def collect_posts(reddit, subreddit_name, post_limit=100):
    """Collect posts with enhanced data collection and sentiment analysis."""
    posts_data = []
    try:
        subreddit = reddit.subreddit(subreddit_name)

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
                    'transformer_score': sentiment_results['transformer_score']
                }

                posts_data.append(post_data)
                logger.info(f"Collected and analyzed post: {post.title[:50]}...")

            except Exception as e:
                logger.warning(f"Error processing post: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error accessing subreddit: {str(e)}")
        raise

    return posts_data

def save_to_csv(posts_data, filename):
    """Save collected posts to CSV file with enhanced error handling."""
    try:
        df = pd.DataFrame(posts_data)

        # Convert timestamp to datetime if not already
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort by timestamp
        df = df.sort_values('timestamp', ascending=False)

        # Save to CSV
        df.to_csv(filename, index=False)
        logger.info(f"Successfully saved {len(posts_data)} posts to {filename}")

        # Generate and log basic statistics
        logger.info(f"Dataset Statistics:")
        logger.info(f"Average sentiment (TextBlob): {df['textblob_sentiment'].mean():.2f}")
        logger.info(f"Most common sentiment (Transformer): {df['transformer_sentiment'].mode()[0]}")
        logger.info(f"Average engagement (score): {df['score'].mean():.2f}")

    except Exception as e:
        logger.error(f"Error saving to CSV: {str(e)}")
        raise

def main():
    try:
        # Initialize Reddit instance
        reddit = initialize_reddit()

        # Define subreddits to analyze
        subreddits = ["ArtificialIntelligence", "ChatGPT", "MachineLearning"]
        all_posts_data = []

        for subreddit in subreddits:
            try:
                logger.info(f"Starting to collect posts from r/{subreddit}...")
                posts_data = collect_posts(reddit, subreddit)
                all_posts_data.extend(posts_data)
            except Exception as e:
                logger.error(f"Error collecting posts from r/{subreddit}: {str(e)}")
                continue  # Continue with next subreddit even if this one fails

        if all_posts_data:  # Only save if we have data
            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reddit_analysis_{timestamp}.csv"
            save_to_csv(all_posts_data, filename)
        else:
            logger.error("No data was collected from any subreddit")

    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        raise

    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()