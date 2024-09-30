# Sentiment_Classification
Sentiment Classification Using Logistic Regression

Here I have implemented naive bayes baseline classifier. Additionally, I am using pytorch to implement a binary logistic regression classifier. Here the main task is sentiment classification for hotel reviews. The input to this model will be a text review, and the output label is a 1 or 0 marking it as positive or negative.

The util.py file provided will load the data and some of the basic modeling. We will install some dependencies required from the provided requirements.txt file.

Task Description : 

We will be implementing and training a logistic regression model for sentiment analysis. Sentiment analysis is the process of determining the sentiment or emotional tone behind a piece of text. The goal is to classify text data as either positive or negative sentiment.

This involves several steps:

- Data Preprocessing: Convert raw text data into a format suitable for machine learning, such as tokenizing the text and extracting features.
- Feature Extraction: Generate numerical features from the text data that can be used as input to the logistic regression model.
- Model Implementation: Define and implement a logistic regression model using a machine learning framework like PyTorch.
- Training: Train the model on a labeled dataset, adjusting the model parameters to minimize the loss function.
- Evaluation: Evaluate the model's performance using metrics such as accuracy, precision, recall, and F1 score.
- Testing: Apply the trained model to a test dataset to assess its generalization performance.

Method used : 

1. Data Preprocessing
Data preprocessing is a crucial step to convert raw text data into a format suitable for machine learning. The preprocessing steps include:

- Tokenization: Splitting the text into individual words or tokens.

- Lowercasing: Converting all characters to lowercase to ensure uniformity.

- Removing Punctuation: Eliminating punctuation marks to focus on the words.

- Removing Stop Words: Removing common words (like "and", "the", etc.) that do not contribute much to the sentiment.

2. Feature Engineering
Feature engineering involves extracting meaningful numerical features from the text data. The features used in this task include:

- Word Count: The number of words in the text.

- Character Count: The total number of characters in the text.

- Presence of Specific Words: Binary features indicating the presence of specific words (e.g., "happy", "sad", "love").

- Punctuation Count: The number of punctuation marks in the text.

- Uppercase Word Count: The number of words in uppercase, which might indicate emphasis.

These features are extracted using custom functions and combined into a feature vector for each text sample.

3. Model Implementation
The logistic regression model is implemented using PyTorch. The model consists of a single linear layer that maps the input features to a single output, which is then passed through a sigmoid function to obtain a probability score.

4. Training
The model is trained using the binary cross-entropy loss function and the Stochastic Gradient Descent (SGD) optimizer. The training loop involves:
- Shuffling the training data and creating batches.

- Forward pass: Computing the model's predictions.

- Loss computation: Calculating the binary cross-entropy loss.

- Backward pass: Computing the gradients.

- Parameter update: Adjusting the model parameters using the optimizer.

5. Evaluation
The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1 score. These metrics provide a comprehensive view of the model's effectiveness in predicting the correct sentiment.

6. Testing
The trained model is applied to a test dataset to assess its generalization performance. The test data is preprocessed and featurized in the same way as the training data. The model's predictions on the test data are compared to the true labels to compute the evaluation metrics.

The data was splitted into 80:20 ratio for training and testing.

This method ensures a systematic approach to building and evaluating a logistic regression model for sentiment analysis, leveraging feature engineering, model training, and performance evaluation.

Usefulness of Sentiment Analysis: 

Sentiment analysis has a wide range of applications in various fields:

- Customer Feedback: Companies can analyze customer reviews and feedback to understand customer satisfaction and identify areas for improvement.
- Social Media Monitoring: Brands can monitor social media platforms to gauge public sentiment about their products or services.
- Market Research: Businesses can analyze sentiment trends to make informed decisions about product launches, marketing strategies, and more.
- Political Analysis: Sentiment analysis can be used to analyze public opinion on political issues, candidates, and policies.
- Content Moderation: Platforms can use sentiment analysis to detect and filter out harmful or inappropriate content.
