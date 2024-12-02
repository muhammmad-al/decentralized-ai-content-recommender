# Privacy-Preserving User Interaction Prediction with Federated Learning

A machine learning system that predicts whether users will interact with social media content while keeping user data private across regions. Built on Reddit data across AI, music, and web3 topics.

## What This Project Does

The system predicts if a user will engage with a post, comparing two approaches:

* Traditional: All user data in one place (95% accuracy)
* Privacy-Preserving: Data stays in local regions (88% accuracy)

The privacy-preserving version only loses 7% accuracy while keeping user data private - showing it's possible to protect user privacy without significantly impacting performance.

## How It Works

### Data Pipeline
* Collects posts through Reddit's PRAW API with automated scraping
* Processes 15,000+ posts across three topic areas
* Runs sentiment analysis to understand post content
* Transforms raw text and metrics into machine learning features

### Model Architecture
* Deep learning model built with TensorFlow
* Natural language processing for text understanding
* Distributed system keeping data private by region
* Performance monitoring and evaluation system

### Key Performance Metrics

| Metric | Centralized | Federated |
|--------|-------------|-----------|
| AUC | 0.9582 | 0.8841 |
| Average Precision | 0.9304 | 0.8369 |
| Interaction Correlation | 0.7114 | 0.6050 |

The federated approach achieves strong performance with only a ~7% reduction in accuracy while providing significant privacy benefits.

The project implements both centralized and federated approaches, demonstrating how privacy-preserving machine learning can work in real-world applications.
