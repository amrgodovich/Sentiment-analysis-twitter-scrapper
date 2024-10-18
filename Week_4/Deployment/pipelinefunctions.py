# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# import string
# from sentence_transformers import SentenceTransformer
# from keras.models import load_model
# from nltk.stem import WordNetLemmatizer
# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# from wordcloud import WordCloud
# import re
# import numpy as np
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# import pickle
# import mlflow.keras
# import mlflow
# import re
# import pandas as pd

# wn = WordNetLemmatizer()
# stopwords_En = stopwords.words('english')
# stopwords_En.remove('no')
# stopwords_En.remove('not')
# nltk.download('punkt')
# nltk.download('punkt_tab')

# mlflow.set_tracking_uri('mlruns')
# mlflow.set_tracking_uri('http://localhost:5000')
# mlflow.set_experiment("Sentiment_analysis_DEPI")

# # loading
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)



# # model_uri = 'runs:/fbd8d80ce1514764b6301ee1058a5e74/neural_network_classifier_model'
# # model = mlflow.pyfunc.load_model(model_uri)
# #------------------------------------------------------------------------------------------------------
# #------------------------------------------------------------------------------------------------------

# # model_bert = SentenceTransformer('all-MiniLM-L6-v2')

# # def load_sentiment_model():
# #     model_path = 'Amazon_review_sent_analysis model.h5'
# #     if not os.path.exists(model_path):
# #         raise FileNotFoundError(f"Model file not found at {model_path}")
# #     model = load_model(model_path)
# #     return model

# #------------------------------------------------------------------------------------------------------
# #------------------------------------------------------------------------------------------------------

# # punctuation
# def remove_punct(text):

#     return "".join([char for char in text if char not in string.punctuation])

# # tokenize text
# def tokenize(text):
#     return word_tokenize(text)

# # remove stopwords
# def remove_stopwords(tokenized_list):

#     return [word for word in tokenized_list if word not in stopwords_En]

# # clean_text function
# def clean_text(text):
    
    
#     # Remove punctuation
#     text = remove_punct(text.lower())
    
#     # Tokenize the text
#     tokens = tokenize(text)
    
#     # Remove stopwords
#     tokens = remove_stopwords(tokens)
    
#     # Lemmatize the tokens
#     text = " ".join([wn.lemmatize(word) for word in tokens])

#     text = remove_non_english(text)
    
#     return text

# def remove_non_english(text):
#     # Convert to string and replace NaN with an empty string
#     text = str(text)
#     text = text.replace('nan', '')  # Replace NaN values with an empty string
#     return re.sub(r'[^a-zA-Z0-9\s]', '', text)

# def preprocess(df):

#     # Clean the text
#     df.text= df.text.astype(str)
#     df['cleaned_Text'] = df['text'].apply(lambda x: clean_text(x))
#     df = df.dropna(subset=['text'])
#     print("Done Cleaning")
#     print(df['cleaned_Text'])

#     return df

# #------------------------------------------------------------------------------------------------------
# #------------------------------------------------------------------------------------------------------
# # model=load_sentiment_model()

# logged_model = 'runs:/f8b0e6e1aee14f4f89c1b68e3e56bfc8/lstm'

# # Load model as a PyFuncModel.
# loaded_model = mlflow.pyfunc.load_model(logged_model)


# def prediction(df):

#     for index, row in df.iterrows():
#         # text = row['cleaned_Text']
#         text = [row['cleaned_Text']]
#         # Get the sentiment label and probability
#         # label, prob = predict_sentiment(text)
#         label, prob = predict_class(text)
#         # Assign the sentiment label and probability to the respective columns
#         df.at[index, 'prediction_label'] = label
#         df.at[index, 'prediction_prob'] = prob

#         df.to_csv('results.csv') # for testing

#     return df


# # def predict_sentiment(text):
# #     # Encode the input text using the BERT model
# #     embedding = model_bert.encode([text])

# #     # Make prediction using the loaded model
# #     prob = model.predict(embedding)[0][0]  # Get the probability score

# #     # Determine the label based on the probability
# #     label = "Positive" if prob > 0.5 else "Negative"

# #     # Return both the label and probability
# #     return label, prob

# def predict_class(text):

#     sentiment_classes = ['Negative', 'Positive']
#     max_len = 50
    
#     # Transforms text to a sequence of integers using a tokenizer object
#     xt = tokenizer.texts_to_sequences(text)
#     # Pad sequences to the same length
#     xt = pad_sequences(xt, padding='post', maxlen=max_len)
    
#     # Do the prediction using the loaded model
#     probabilities = loaded_model.predict(xt)
    
#     # Get the predicted class (with highest probability)
#     predicted_class = probabilities.argmax(axis=1)[0]
    
#     # Print the predicted sentiment and the corresponding probability
#     predicted_sentiment = sentiment_classes[predicted_class]
#     predicted_probability = probabilities[0][predicted_class]
    
#     # print(f"The predicted sentiment is '{predicted_sentiment}' with probability {predicted_probability:.4f}")
    
#     return predicted_sentiment, predicted_probability

# #------------------------------------------------------------------------------------------------------
# #------------------------------------------------------------------------------------------------------


# def visualize(df, base_folder):
#     Visualizer(df).get_high_score(base_folder)
#     Visualizer(df).get_wordcloud(base_folder)
#     Visualizer(df).percentage(base_folder)
#     Visualizer(df).trend(base_folder)

# #------------------------------------------------------------------------------------------------------

# class Visualizer:
#     def __init__(self, df):
#         self.df = df
#         self.stop_words = set(stopwords.words('english'))

#     def trend(self, base_folder):
#         self.df['time'] = pd.to_datetime(self.df['time'], format='%Y-%m-%dT%H:%M:%S.%f%z')
#         self.df = self.df.sort_values(by='time')

#         # Resample data to every 12 hours and count positive and negative occurrences
#         df_resampled = self.df.resample('12H', on='time')['prediction_label'].value_counts().unstack().fillna(0)

#         # plt.figure(figsize=(10, 6))
#         try:
#             plt.plot(df_resampled.index, df_resampled['Positive'], marker='o', color='g', linestyle='-', label='Positive')
#         except KeyError:
#             print("Warning: No 'Positive' column found in the resampled data.")

#         try:
#             plt.plot(df_resampled.index, df_resampled['Negative'], marker='o', color='r', linestyle='-', label='Negative')
#         except KeyError:
#             print("Warning: No 'Negative' column found in the resampled data.")

#         plt.xlabel('Time')
#         plt.ylabel('Frequency')
#         plt.title('Positive and Negative Trends Over Time (12-hour intervals)')
#         plt.legend()
#         plt.grid(True)
#         plt.xticks(rotation=45)
#         plt.tight_layout()

#         # Save the plot
#         trend_plot_path = os.path.join(base_folder, 'trend_plot.png')
#         plt.savefig(trend_plot_path)
#         plt.close()

#         print(f"Trend plot saved at {trend_plot_path}")

#     def get_wordcloud(self, base_folder):
#         positive_tweets = self.df[self.df['prediction_label'] == 'Positive']['cleaned_Text'].tolist()
#         negative_tweets = self.df[self.df['prediction_label'] == 'Negative']['cleaned_Text'].tolist()

#         positive_text = " ".join(positive_tweets)
#         negative_text = " ".join(negative_tweets)

#         # Generate word clouds
#         if positive_text:
#             try:
#                 positive_wordcloud = WordCloud(width=800, height=800, background_color='white', 
#                                             stopwords=self.stop_words, min_font_size=10).generate(positive_text)
#                 positive_wc_path = os.path.join(base_folder, 'positive_wordcloud.png')
#                 self._plot_wordcloud(positive_wordcloud, 'Positive Tweets', positive_wc_path)
#                 print(f"Positive word cloud saved at {positive_wc_path}")
#             except Exception as e:
#                 print(f"Error creating positive word cloud: {e}")
#         else:
#             print("No positive words to create a cloud.")

#         if negative_text:
#             try:
#                 negative_wordcloud = WordCloud(width=800, height=800, background_color='white', 
#                                             stopwords=self.stop_words, min_font_size=10).generate(negative_text)
#                 negative_wc_path = os.path.join(base_folder, 'negative_wordcloud.png')
#                 self._plot_wordcloud(negative_wordcloud, 'Negative Tweets', negative_wc_path)
#                 print(f"Negative word cloud saved at {negative_wc_path}")
#             except Exception as e:
#                 print(f"Error creating negative word cloud: {e}")
#         else:
#             print("No negative words to create a cloud.")

#             print(f"Word clouds saved at {base_folder}")

#     def percentage(self, base_folder):
#         sentiment_counts = self.df['prediction_label'].value_counts()

#         # plt.figure(figsize=(5, 5))
#         plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
#         plt.title('Sentiment Distribution')

#         # Save the pie chart
#         percentage_plot_path = os.path.join(base_folder, 'sentiment_percentage.png')
#         plt.savefig(percentage_plot_path)
#         plt.close()

#         print(f"Percentage plot saved at {percentage_plot_path}")

#     def get_high_score(self, base_folder):
#         sorted_data = self.df.sort_values(by='prediction_prob', ascending=False)
#         positive_top2 = sorted_data[sorted_data['prediction_label'] == 'Positive'].head(2)['text'].tolist()
#         negative_top2 = sorted_data[sorted_data['prediction_label'] == 'Negative'].tail(2)['text'].tolist()

#         # Save the top positive and negative tweets to text files
#         positive_file_path = os.path.join(base_folder, 'positive_top2.txt')
#         negative_file_path = os.path.join(base_folder, 'negative_top2.txt')

#         with open(positive_file_path, 'w') as pos_file:
#             for tweet in positive_top2:
#                 pos_file.write(f"{tweet}\n")

#         with open(negative_file_path, 'w') as neg_file:
#             for tweet in negative_top2:
#                 neg_file.write(f"{tweet}\n")

#         print(f"Top tweets saved at {base_folder}")

#     def _plot_wordcloud(self, wordcloud, title, filepath):
#         # plt.figure(figsize=(10, 5))
#         plt.imshow(wordcloud, interpolation='bilinear')
#         plt.title(title, pad=20.0)
#         plt.axis('off')

#         # Save word cloud plot
#         plt.savefig(filepath)
#         plt.close()


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import os
from wordcloud import WordCloud
import re
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import mlflow
import mlflow.keras

# Initialization
nltk.download('punkt')
nltk.download('wordnet')
wn = nltk.WordNetLemmatizer()
stopwords_En = set(stopwords.words('english'))
stopwords_En -= {'no', 'not'}

# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# MLflow setup
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment("Sentiment_analysis_DEPI")

# Load model
logged_model = 'runs:/f8b0e6e1aee14f4f89c1b68e3e56bfc8/lstm'
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Text processing functions
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower()) 
    tokens = word_tokenize(text)
    tokens = [wn.lemmatize(word) for word in tokens if word not in stopwords_En]
    return " ".join(tokens)

def preprocess(df):
    df = df[df['user'].str.startswith('@')]
    df['cleaned_Text'] = df['text'].astype(str).apply(clean_text)
    df.dropna(subset=['text'], inplace=True)
    return df

# Prediction
def predict_class(text):
    sentiment_classes = ['Negative', 'Positive']
    max_len = 50
    xt = pad_sequences(tokenizer.texts_to_sequences([text]), padding='post', maxlen=max_len)
    probabilities = loaded_model.predict(xt)
    predicted_class = probabilities.argmax(axis=1)[0]
    return sentiment_classes[predicted_class], probabilities[0][predicted_class]

def prediction(df):
    df[['prediction_label', 'prediction_prob']] = df['cleaned_Text'].apply(
        lambda x: pd.Series(predict_class(x)))
    df.to_csv('results.csv', index=False)  # now for testing only
    return df


def visualize(df):
    visualizer = Visualizer(df)  # Instantiate the Visualizer with the DataFrame
    visualizer.trend()            # Call trend method
    visualizer.get_high_score()   # Call method to get high score tweets
    visualizer.get_wordcloud()     # Call method to generate word clouds
    visualizer.percentage()        # Call method for percentage visualization


    
# Visualization
class Visualizer:
    def __init__(self, df):
        self.df = df
        self.stop_words = set(stopwords.words('english'))

    def trend(self):
        self.df['time'] = pd.to_datetime(self.df['time'], format='%Y-%m-%dT%H:%M:%S.%f%z')
        df_resampled = self.df.resample('48H', on='time')['prediction_label'].value_counts().unstack().fillna(0)
        plt.plot(df_resampled.index, df_resampled.get('Positive', []), marker='o', color='g', label='Positive')
        plt.plot(df_resampled.index, df_resampled.get('Negative', []), marker='o', color='r', label='Negative')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.title('Sentiment Trends (two days intervals)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        trend_plot_path = os.path.join('static', 'trend_plot.png')
        plt.savefig(trend_plot_path)
        plt.close()

    def get_wordcloud(self):
        for label in ['Positive', 'Negative']:
            text = " ".join(self.df[self.df['prediction_label'] == label]['cleaned_Text'].tolist())
            if text:
                wc = WordCloud(width=800, height=800, background_color='white', stopwords=self.stop_words).generate(text)
                wc_path = os.path.join('static', f'{label.lower()}_wordcloud.png')
                self._plot_wordcloud(wc, f'{label} Tweets', wc_path)

    def percentage(self):
        sentiment_counts = self.df['prediction_label'].value_counts()
        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
        plt.title('Sentiment Distribution')
        percentage_plot_path = os.path.join('static', 'sentiment_percentage.png')
        plt.savefig(percentage_plot_path)
        plt.close()

    def get_high_score(self):
        sorted_data = self.df.sort_values(by='prediction_prob', ascending=False)
        for label in ['Positive', 'Negative']:
            tweets = sorted_data[sorted_data['prediction_label'] == label].head(2)['text'].tolist()
            with open(os.path.join('static', f'{label.lower()}_top2.txt'), 'w') as f:
                f.write("\n".join(tweets))

    def _plot_wordcloud(self, wordcloud, title, filepath):
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title, pad=20.0)
        plt.axis('off')
        plt.savefig(filepath)
        plt.close()

