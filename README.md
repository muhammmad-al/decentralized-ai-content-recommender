# Decentralized AI Content Recommender

A decentralized federated learning framework for Reddit content recommendation. This system uses multiple nodes to train on different content categories while maintaining data privacy and improving overall recommendation accuracy.

## Project Progress

### 1. Data Collection
- Successfully collected Reddit data from multiple subreddits across 3 categories:
  - **AI**: r/ChatGPT, r/artificial, r/MachineLearning
  - **Music**: r/hiphopheads, r/WeAreTheMusicMakers, r/edmproduction
  - **Web3**: r/cryptocurrency, r/ethereum, r/CryptoTechnology, r/defi, r/NFT

- Collected features:
  - Post titles and content
  - Engagement metrics (scores, comments)
  - Timestamps and metadata

### 2. Feature Engineering
- Implemented text preprocessing and cleaning
- Added sentiment analysis using:
  - TextBlob for polarity analysis
  - Transformer-based sentiment scoring
  - Combined sentiment features
- Created derived features:
  - Log-transformed engagement metrics
  - Text length and complexity metrics
  - Sentiment magnitude and compounds

### 3. Model Architecture
Current implementation:
- TF-IDF vectorization (300 features)
- Neural network with:
  - Input layer: 128 neurons, ReLU, L2 regularization
  - Hidden layer: 64 neurons, ReLU, L2 regularization
  - Batch normalization and dropout layers
  - Output: 3-way classification (AI, Music, Web3)

### 4. Current Results
Single node performance:
- Training Accuracy: 92.70%
- Validation Accuracy: 87.89%
- Test Accuracy: 87.25%
