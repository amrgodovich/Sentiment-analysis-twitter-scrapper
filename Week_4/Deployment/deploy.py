from flask import Flask, render_template, request
from pipelinefunctions import preprocess, prediction, visualize
from web_scraping import scrape_tweets
import os
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        topic = request.form['topic']
        username = request.form['username']
        password = request.form['password']
        max_tweets = 50

        df = scrape_tweets(username, password, topic, max_tweets)
        # df = pd.read_csv("scraped_data/tweets_ronaldo_v1/tweets_ronaldo_v1.csv")

        if df is None or df.empty:
            return "No tweets were scraped."

        df = preprocess(df)
        df = prediction(df)
        
        # Visualize and save paths to images/text
        visualize(df)

        # Static paths for visualization outputs
        positive_wc_path = 'positive_wordcloud.png'
        negative_wc_path = 'negative_wordcloud.png'
        trend_path = 'trend_plot.png'
        pie_chart_path = 'sentiment_percentage.png'

        # Read top tweets from text files
        positive_top2 = _read_top_tweets('positive_top2.txt')
        negative_top2 = _read_top_tweets('negative_top2.txt')

        # Render results page
        return render_template(
            'results.html',
            topic=topic,
            positive_wc_path=positive_wc_path,
            negative_wc_path=negative_wc_path,
            trend_path=trend_path,
            pie_chart_path=pie_chart_path,
            positive_top2=positive_top2,
            negative_top2=negative_top2
        )

    return render_template('index.html')

def _read_top_tweets(filename):
    file_path = os.path.join('static', filename)
    with open(file_path, 'r') as file:
        tweets = file.readlines()
    return [tweet.strip() for tweet in tweets]

if __name__ == '__main__':
    app.run(debug=True, port=8080)