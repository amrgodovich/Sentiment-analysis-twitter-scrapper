import time
import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def scrape_tweets(username, password, topic, max_tweets, folder='scraped_data'):
    def scroll_down(browser):
        browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

    def close_popup(browser):
        try:
            close_button = browser.find_element(By.XPATH, '//button[@data-testid="xMigrationBottomBar"]')
            close_button.click()
            time.sleep(3)
        except Exception as e:
            print(f"No popup to close: {e}")

    # Ensure the main folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    options = webdriver.FirefoxOptions()
    # options.add_argument('--headless')

    with webdriver.Firefox(options=options) as browser:
        url = 'https://twitter.com/'
        browser.get(url)

        wait = WebDriverWait(browser, 90)

        close_popup(browser)

        login_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//a[@href="/login"]')))
        login_button.click()

        username_input = wait.until(EC.presence_of_element_located((By.XPATH, './/input[@name="text"]')))
        username_input.send_keys(username)
        username_input.send_keys(Keys.RETURN)

        time.sleep(3)

        try:
            password_input = wait.until(EC.presence_of_element_located((By.XPATH, '//input[@type="password"]')))
            password_input.send_keys(password)
            password_input.send_keys(Keys.RETURN)
        except Exception as e:
            print("Error locating password input:", e)
            return None, None

        # Wait for the search input to appear
        try:
            search_input = wait.until(EC.presence_of_element_located((By.XPATH, '//input[@aria-label="Search query"]')))
            search_input.send_keys(topic)
            search_input.send_keys(Keys.RETURN)
        except Exception as e:
            print("Error locating search input:", e)
            return None, None

        current_tweets = 0
        user_data = []
        text_data = []
        time_data = []

        while current_tweets < max_tweets:
            for _ in range(5):
                scroll_down(browser)

            tweets = wait.until(EC.presence_of_all_elements_located((By.XPATH, '//article[@role="article"]')))

            for tweet in tweets:
                try:
                    user = tweet.find_element(By.XPATH, './/span[contains(text(), "@")]').text
                    text = tweet.find_element(By.XPATH, ".//div[@lang]").text
                    tweet_time = tweet.find_element(By.XPATH, ".//time").get_attribute("datetime")

                    tweets_data = [user, text, tweet_time]
                except Exception as e:
                    print(f"Error extracting tweet: {e}")
                    tweets_data = ['user', 'text', "time"]

                user_data.append(tweets_data[0])
                text_data.append(" ".join(tweets_data[1].split()))
                time_data.append(tweets_data[2])

                current_tweets += 1

            print(f"Scraped {current_tweets} tweets")

            if current_tweets >= max_tweets:
                break

        # # Create a versioned subdirectory for the topic
        # base_folder = os.path.join(folder, f'tweets_{topic.replace(" ", "_")}_v1')
        
        # # Check for version increment if folder already exists
        # version = 1
        # while os.path.exists(base_folder):
        #     version += 1
        #     base_folder = os.path.join(folder, f'tweets_{topic.replace(" ", "_")}_v{version}')

        # os.makedirs(base_folder)
        base_folder='static'

        # Set the filename based on the topic and version
        # filename = f'tweets_{topic.replace(" ", "_")}_v{version}.csv'
        filename = f'tweets_{topic.replace(" ", "_")}.csv'
        filepath = os.path.join(base_folder, filename)

        df = pd.DataFrame({'user': user_data, 'text': text_data, 'time': time_data})
        df.to_csv(filepath, index=False)
        print(f"Total {current_tweets} tweets scraped. Data saved to {filepath}")

    return df