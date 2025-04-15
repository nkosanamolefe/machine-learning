# Airbnb Reviews Sentiment Analysis

In the fiercely competitive hospitality landscape, understanding and enhancing guest satisfaction is paramount for hosts, property managers, and platform administrators alike. Maintaining high occupancy rates and garnering positive reviews are crucial for success in this dynamic industry.

Airbnb, a dominant force in the short-term rental market, offers a treasure trove of data through guest reviews. This project harnesses the power of RoBERTa, a state-of-the-art language model from HuggingFace's Transformers library, to perform sentiment analysis on this valuable review data

## Problem Overview

The central objective of this project is to construct a robust sentiment analysis pipeline capable of gauging guest satisfaction over time. By meticulously analyzing the sentiment expressed in Airbnb reviews, this system empowers hosts and property managers to:

- **Track changes in guest satisfaction:** Identify trends and patterns in sentiment to understand how guest experiences are evolving.
- **Evaluate the effectiveness of improvements:** Measure the impact of changes made to properties or services on guest sentiment.
- **Make data-driven decisions:** Leverage sentiment insights to inform strategic decisions aimed at enhancing guest experiences.

## Data

This comprehensive [dataset](https://data.insideairbnb.com/south-africa/wc/cape-town/2024-12-28/data/reviews.csv.gz) from [Inside Airbnb](http://insideairbnb.com/), comprising over `500,000 rows`, offers a broad perspective on guest experiences in Cape Town, South Africa. The dataset snapshot used for this analysis was collected on December 28, 2024.

The dataset comprises several key features that are crucial for understanding guest sentiment and experiences:

- **listing_id:** A unique identifier for each Airbnb listing.
- **reviewer_name:** The name of the guest who wrote the review.
- **comments:** The text content of the guest's review.
- **date:** The date when the review was submitted.

## Methodology

### Handling Missing Values

Data cleaning steps:

- A throughout inspection of the dataset revealed the presence of missing values.
  - 121 missing values in the `comments` column
  - 1 missing value in the `reviewer_name` column.
- To address the issue of missing data, rows containing missing values in either the 'comments' or 'reviewer_name' columns were removed from the dataset using the `df.dropna()` function.

### Preprocessing

Given the extensive dataset of over 500,000 guest reviews, processing all of them would be computationally intensive. This project strategically focuses on a single listing with more than 20 reviews (can be adjusted as required) for analysis.

- Once a random listing is selected, a language detection with `langdetect` is used to filter out non-English reviews, thereby optimizing the dataset for English-specific RoBERTa sentiment analysis. Standard RoBERTa models are primarily trained on English text. Feeding them non-English reviews would likely result in poor and unreliable sentiment predictions.
- Clean review text by removing unnecessary characters, HTML tags, URLs, and special symbols.

### Sentiment Analysis with RoBERTa

This project leverages the power of the RoBERTa (Robustly Optimized BERT Pretraining Approach) model from HuggingFace's Transformers library for sentiment analysis of Airbnb guest reviews. RoBERTa is a state-of-the-art natural language processing model known for its exceptional performance in various text classification tasks.

Here's how RoBERTa was integrated into the sentiment analysis pipeline:

1. Model Loading:

   - The pre-trained cardiffnlp/twitter-roberta-base-sentiment model and its corresponding tokenizer were loaded from the Transformers library using AutoModelForSequenceClassification and AutoTokenizer. This model is specifically fine-tuned for sentiment analysis on Twitter data, making it well-suited for analyzing guest reviews as well.
2. Tokenization:

   - Guest review texts were processed using the RoBERTa tokenizer to convert them into numerical representations that the model can understand. This involved breaking down the text into individual words or subwords (tokens) and assigning a unique numerical ID to each token.
3. Sentiment Classification:

   - The tokenized review text was then fed into the RoBERTa model for sentiment classification. The model assigns probabilities to three sentiment categories: positive, negative, and neutral.
4. Sentiment Scores:

   - The output probabilities from the RoBERTa model were converted into sentiment scores using the softmax function. These scores represent the model's confidence in each sentiment category for a given review. The scores were stored in a dictionary format with keys 'roberta_neg', 'roberta_neu', and 'roberta_pos' and subsequently added to the main DataFrame for further analysis.

By leveraging RoBERTa's advanced language understanding capabilities, the project was able to effectively analyze guest sentiment expressed in Airbnb reviews, providing valuable insights into guest satisfaction trends over time. I hope this explanation clarifies the use of RoBERTa for sentiment analysis in your project. Let me know if you have any other questions.

## Results and Visualization

## Conclusion