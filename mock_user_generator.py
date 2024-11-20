import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from collections import defaultdict
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockUserInteractionGenerator:
    def __init__(self, num_users=1000, interaction_sparsity=0.01):
        """
        Initialize the mock user interaction generator
        
        Args:
            num_users: Number of synthetic users to create
            interaction_sparsity: Fraction of total possible interactions to generate
        """
        self.num_users = num_users
        self.interaction_sparsity = interaction_sparsity
        self.user_preferences = self._generate_user_preferences()
        
    def _generate_user_preferences(self):
        """Generate synthetic user preference profiles"""
        preferences = {}
        
        # Define possible user archetypes
        archetypes = {
            'ai_enthusiast': {'ai': 0.7, 'web3': 0.2, 'music': 0.1},
            'music_lover': {'music': 0.7, 'ai': 0.2, 'web3': 0.1},
            'crypto_trader': {'web3': 0.7, 'ai': 0.2, 'music': 0.1},
            'tech_generalist': {'ai': 0.4, 'web3': 0.4, 'music': 0.2},
            'casual_browser': {'music': 0.4, 'ai': 0.3, 'web3': 0.3}
        }
        
        logger.info("Generating user archetypes...")
        # Assign users to archetypes with some random variation
        for user_id in range(self.num_users):
            archetype = random.choice(list(archetypes.keys()))
            base_prefs = archetypes[archetype].copy()
            
            # Add random noise to preferences
            noise = np.random.normal(0, 0.1, len(base_prefs))
            preferences[user_id] = {
                k: max(0.01, min(0.99, v + noise[i]))
                for i, (k, v) in enumerate(base_prefs.items())
            }
            
        return preferences

    def generate_interactions(self, content_df):
        """Generate synthetic user interactions with content"""
        interactions = []
        
        # Calculate number of interactions to generate
        total_possible = len(content_df) * self.num_users
        num_interactions = int(total_possible * self.interaction_sparsity)
        
        logger.info(f"Generating {num_interactions} interactions...")
        
        # Track user interaction history
        user_history = defaultdict(set)
        
        # Generate base timestamp
        base_timestamp = datetime.now() - timedelta(days=30)
        
        for i in range(num_interactions):
            if i % 10000 == 0:  # Progress update every 10000 interactions
                logger.info(f"Generated {i}/{num_interactions} interactions...")
                
            user_id = random.randint(0, self.num_users - 1)
            
            # Select content based on user preferences
            category_choice = random.choices(
                list(self.user_preferences[user_id].keys()),
                list(self.user_preferences[user_id].values())
            )[0]
            
            # Filter for chosen category and exclude already interacted content
            available_content = content_df[
                (content_df['category'] == category_choice) & 
                ~(content_df.index.isin(user_history[user_id]))
            ]
            
            if len(available_content) == 0:
                continue
                
            # Select content and generate interaction
            content = available_content.sample(1).iloc[0]
            
            # Generate interaction strength based on user preference
            base_strength = self.user_preferences[user_id][category_choice]
            interaction_strength = max(0.1, min(1.0, 
                np.random.normal(base_strength, 0.2)))
            
            # Generate timestamp with some randomness
            timestamp = base_timestamp + timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            # Generate interaction type and engagement metrics
            interaction_type = random.choices(
                ['view', 'like', 'comment', 'share'],
                weights=[0.7, 0.2, 0.08, 0.02]
            )[0]
            
            engagement_duration = 0
            if interaction_type != 'view':
                engagement_duration = int(np.random.exponential(300))  # mean 5 minutes
            
            interaction = {
                'user_id': user_id,
                'content_id': content.name,
                'category': category_choice,
                'interaction_type': interaction_type,
                'interaction_strength': interaction_strength,
                'engagement_duration': engagement_duration,
                'timestamp': timestamp
            }
            
            interactions.append(interaction)
            user_history[user_id].add(content.name)
        
        # Convert to DataFrame
        interactions_df = pd.DataFrame(interactions)
        
        # Add some derived features
        interactions_df['day_of_week'] = interactions_df['timestamp'].dt.dayofweek
        interactions_df['hour_of_day'] = interactions_df['timestamp'].dt.hour
        
        return interactions_df

    def generate_user_features(self):
        """Generate synthetic user feature profiles"""
        logger.info("Generating user features...")
        user_features = []
        
        for user_id in range(self.num_users):
            # Basic user features
            activity_level = np.random.normal(0.5, 0.2)
            feature_vector = {
                'user_id': user_id,
                'activity_level': max(0.1, min(1.0, activity_level)),
                'account_age_days': random.randint(1, 1000),
                'is_active': random.random() > 0.1  # 90% active users
            }
            
            # Add category preferences
            for category, pref in self.user_preferences[user_id].items():
                feature_vector[f'{category}_preference'] = pref
            
            user_features.append(feature_vector)
            
        return pd.DataFrame(user_features)

def create_mock_interaction_dataset(content_filepath_pattern="data/raw/reddit_analysis_{}_5000_posts.csv"):
    """Create a complete mock dataset including content, users, and interactions"""
    logger.info("Starting mock data generation...")
    
    # Load and combine content
    categories = ['ai', 'music', 'web3']
    dfs = []
    
    logger.info("Loading content data from CSV files...")
    for category in categories:
        filepath = content_filepath_pattern.format(category)
        if not os.path.exists(filepath):
            logger.error(f"Content file not found: {filepath}")
            raise FileNotFoundError(f"Missing content file: {filepath}")
            
        df = pd.read_csv(filepath)
        df['category'] = category
        dfs.append(df)
        logger.info(f"Loaded {len(df)} posts for {category}")
    
    combined_content_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined content dataset: {len(combined_content_df)} posts")
    
    # Initialize generator
    logger.info("Generating synthetic user profiles...")
    generator = MockUserInteractionGenerator(num_users=1000, interaction_sparsity=0.01)
    
    # Generate interactions
    logger.info("Generating user interactions...")
    interactions_df = generator.generate_interactions(combined_content_df)
    logger.info(f"Generated {len(interactions_df)} interactions")
    
    # Generate user features
    logger.info("Generating user features...")
    user_features_df = generator.generate_user_features()
    logger.info(f"Generated features for {len(user_features_df)} users")
    
    # Save mock data
    logger.info("Saving generated data...")
    os.makedirs('data/mock', exist_ok=True)
    interactions_df.to_csv('data/mock/user_interactions.csv', index=False)
    user_features_df.to_csv('data/mock/user_features.csv', index=False)
    
    logger.info("\nMock Dataset Summary:")
    logger.info(f"Number of users: {len(user_features_df)}")
    logger.info(f"Number of interactions: {len(interactions_df)}")
    logger.info(f"Number of content items: {len(combined_content_df)}")
    logger.info("\nInteraction types distribution:")
    logger.info(interactions_df['interaction_type'].value_counts(normalize=True))
    
    return interactions_df, user_features_df, combined_content_df

if __name__ == "__main__":
    try:
        interactions_df, user_features_df, content_df = create_mock_interaction_dataset()
        logger.info("Mock data generation completed successfully!")
    except Exception as e:
        logger.error(f"Error generating mock data: {str(e)}")