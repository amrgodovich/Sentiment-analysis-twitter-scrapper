# Customer Sentiment and Trend Analysis Web App

This repository contains a **Customer Sentiment and Trend Analysis** web app that allows users to input a topic and automatically scrape tweets related to that topic. The app processes the scraped data, performs sentiment analysis, and visualizes the results using various tools like word clouds, pie charts, and trend plots.

## Features

- **Twitter Scraping**: The app uses Selenium to scrape tweets based on a user-provided topic.
- **Data Preprocessing**: Cleans and preprocesses the tweets to remove noise, including special characters, stop words, and more.
- **Sentiment Analysis**: A pre-trained sentiment analysis model predicts whether each tweet has positive or negative sentiment.
- **Data Visualization**: Visualizes sentiment distribution, top words in the dataset (word cloud), and high-confidence tweets for both positive and negative sentiments.

## Tech Stack

- **Flask**: Backend framework for web app development.
- **Selenium**: For web scraping Twitter data.
- **TensorFlow**: Pre-trained sentiment analysis model.
- **Matplotlib & WordCloud**: Libraries for visualizing sentiment results.
- **HTML/CSS**: Frontend template for displaying results.
- **MLflow**: frame works to enable MLops techinques and enhance the integration/deployment process.

## Pipeline

1. **Input**: User inputs a topic, username, and password via a web form.
2. **Scraping**: Selenium scrapes recent tweets related to the input topic.
3. **Preprocessing**: The scraped data is cleaned and prepared for analysis using custom preprocessing functions.
4. **Prediction**: A TensorFlow model predicts the sentiment of each tweet (positive/negative).
5. **Visualization**: The results are visualized, including:
   - Sentiment distribution (pie chart)
   - Trends over time (line plot)
   - Word cloud of common words
   - Display of top tweets based on sentiment scores

## Datasets
1. **Twitter Dataset:** [Twitter and Reddit Sentimental analysis Dataset](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset)
2. **Amazon Dataset:** [Amazon Product Reviews](https://www.kaggle.com/datasets/mahmudulhaqueshawon/amazon-product-reviews)


## Installation

1. **Clone the repository:**

   ```bash
   git clone https://https://github.com/amrgodovich/Sentiment-analysis-twitter-scrapper.git
   cd week_4/Deployment

1. **Running MLflow**

   ```bash
    mlflow ui
   
2. **Running the project (in a separate terminal)**

  ```bash
    python deploy.py
