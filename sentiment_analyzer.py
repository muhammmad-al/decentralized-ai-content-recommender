import pandas as pd
import numpy as np
from textblob import TextBlob
from transformers import pipeline
import logging
from tqdm import tqdm
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def perform_sentiment_analysis(text):
    """Perform sentiment analysis using both TextBlob and Transformers"""
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
        transformer_result = sentiment_analyzer(text[:512])[0]

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

def process_file(filename):
    """Process a single CSV file"""
    logger.info(f"Processing {filename}")
    
    # Read CSV
    df = pd.read_csv(filename)
    total_rows = len(df)
    
    # Initialize sentiment columns
    df['textblob_sentiment'] = 0.0
    df['transformer_sentiment'] = 'NEUTRAL'
    df['transformer_score'] = 0.5
    
    # Process each row
    for idx in tqdm(range(total_rows), desc="Analyzing sentiments"):
        # Combine title and text for analysis
        combined_text = f"{df.loc[idx, 'title']} {df.loc[idx, 'cleaned_text']}"
        
        # Get sentiment
        sentiment_results = perform_sentiment_analysis(combined_text)
        
        # Update DataFrame
        df.loc[idx, 'textblob_sentiment'] = sentiment_results['textblob_sentiment']
        df.loc[idx, 'transformer_sentiment'] = sentiment_results['transformer_sentiment']
        df.loc[idx, 'transformer_score'] = sentiment_results['transformer_score']
        
        # Save progress every 100 rows in case of interruption
        if (idx + 1) % 100 == 0:
            df.to_csv(filename.replace('.csv', '_with_sentiment.csv'), index=False)
    
    # Calculate combined sentiment score
    df['sentiment_compound'] = (df['textblob_sentiment'] + 
                              (df['transformer_score'] - 0.5) * 2) / 2
    
    # Save final results
    output_file = filename.replace('.csv', '_with_sentiment.csv')
    df.to_csv(output_file, index=False)
    
    # Log statistics
    logger.info(f"\nSentiment Statistics for {filename}:")
    logger.info(f"Average TextBlob sentiment: {df['textblob_sentiment'].mean():.3f}")
    logger.info(f"Average Transformer score: {df['transformer_score'].mean():.3f}")
    logger.info(f"Most common Transformer sentiment: {df['transformer_sentiment'].mode().iloc[0]}")
    
    return df

def main():
    # Initialize sentiment analyzer
    global sentiment_analyzer
    logger.info("Initializing sentiment analyzer...")
    sentiment_analyzer = pipeline("sentiment-analysis", 
                                model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Process each category
    categories = ['ai', 'music', 'web3']
    
    for category in categories:
        input_file = f'data/raw/reddit_analysis_{category}_5000_posts.csv'
        
        if not os.path.exists(input_file):
            logger.warning(f"File not found: {input_file}")
            continue
            
        try:
            df = process_file(input_file)
            logger.info(f"Successfully processed {category} category\n")
        except Exception as e:
            logger.error(f"Error processing {category}: {str(e)}")

if __name__ == "__main__":
    main()