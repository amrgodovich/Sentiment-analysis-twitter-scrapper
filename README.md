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

## Project Structure

   ```bash
   ├── static/                   # Folder for storing scraped tweet data and visualizations
   ├── templates/                # HTML templates for the web app
   │   ├── index.html            # Main page for topic input
   │   └── results.html          # Results page displaying visualizations
   ├── pipelinefunctions.py      # Module containing Preprocess, Prediction, and Visualizer classes
   ├── web_scraping.py           # web scrapper (from X) app
   ├── deploy.py                 # Main Flask app
   ```

## Datasets
1. **Twitter Dataset:** [Twitter and Reddit Sentimental analysis Dataset](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset)
2. **Amazon Dataset:** [Amazon Product Reviews](https://www.kaggle.com/datasets/mahmudulhaqueshawon/amazon-product-reviews)

## Screenshots
- **Input**:
  ![image](https://github.com/user-attachments/assets/35f04b60-49f6-47d8-9c37-464e7b7fd81f)


- **Visualizations**:
  
   ![image](https://github.com/user-attachments/assets/b68c5ebb-06db-468c-950c-f74f2935f05d)
   ![image](https://github.com/user-attachments/assets/9f79cd26-9683-43f3-a06e-4140dc090ea9)
   ![image](https://github.com/user-attachments/assets/3745931b-1487-4ce2-8f77-1b8740ea389d)
   ![image](https://github.com/user-attachments/assets/0e9a4665-0952-46ca-98ce-f8fbe837f352)
   ![image](https://github.com/user-attachments/assets/cb43ca6b-a80d-42f0-9c36-aae836e5128b)


- **MLflow**:
  ![Opera Snapshot_2024-10-18_183202_127 0 0 1](https://github.com/user-attachments/assets/c17e444a-a731-45c9-8761-25692de964b1)


## Installation

1. **Clone the repository:**

   ```bash
   git clone https://https://github.com/amrgodovich/Sentiment-analysis-twitter-scrapper.git
   cd week_4/Deployment

2. **Running MLflow**

   ```bash
    mlflow ui
   
3. **Running the project (in a separate terminal)**

  ```bash
    python deploy.py
   ```

## Team Members:

1.**Amr Essam:** [LinkedIn](https://www.linkedin.com/in/amrgodovich/)

2.**Hadeel Wael:** [LinkedIn](https://www.linkedin.com/in/hadeel-wael-014097253/)

3.**A'laa Abdelhay:** [LinkedIn](https://www.linkedin.com/in/a-laa-abdelhay-16a909239/)

4.**Ebtehal Karam:** [LinkedIn](https://www.linkedin.com/in/ebtehal-karam-197939267/)

5.**Yousef Wael:** 
