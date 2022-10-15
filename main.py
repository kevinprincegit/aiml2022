from flask import Flask

from flask_restful import Api, Resource
import pandas as pd
import re
import os
import tweepy
import requests
import json
import csv
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import joblib
import warnings

warnings.filterwarnings('ignore')

APP = Flask(__name__)
API = Api(APP)

#nltk.download('stopwords')
port_stem = PorterStemmer()


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


m_jlib = joblib.load('model/predict.mdl')
transformer = joblib.load('model/transform.mdl')
vectorizer = joblib.load('model/vectorizer.mdl')

os.environ[
    'TOKEN'] = 'AAAAAAAAAAAAAAAAAAAAAKcgggEAAAAA4L5jgM2FPWh69QDHbyH9be4E%2BKU%3DORwQF1WPicLs4G8pjhehGvmCaPLFu9SZpkCPTSz3C8xd8Trwqu'


def auth():
    return os.getenv('TOKEN')


def create_headers_(bearer_token):
    headers_ = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers_


def create_url(key, max_res):
    search_url = "https://api.twitter.com/2/tweets/search/recent"  # Change to the endpoint you want to collect data from

    # change params based on the endpoint you are using
    query_params = {'query': key,
                    'max_results': max_res,
                    'expansions': 'referenced_tweets.id.author_id',

                    }
    return search_url, query_params


def append_to_csv(json_response, fileName):
    # A counter variable
    counter = 0

    # Open OR create the target CSV file
    csvFile = open(fileName, "a", newline="", encoding='utf-8')
    csvWriter = csv.writer(csvFile)

    # Loop through each tweet
    for tweet in json_response['includes']['tweets']:
        content = ' '.join(re.sub("\W", " ", tweet["text"]).split())
        content.lower()
        emoji_pattern.sub(r'', content)
        user = tweet["author_id"]
        res = [content, user]
        csvWriter.writerow(res)
        counter += 1
    csvFile.close()

    # Print the number of tweets for this iteration
    print("# of Tweets added from this response: ", counter)


emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002500-\U00002BEF"  # chinese char
                           u"\U00002702-\U000027B0"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U00010000-\U0010ffff"
                           u"\u2640-\u2642"
                           u"\u2600-\u2B55"
                           u"\u200d"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"  # dingbats
                           u"\u3030"
                           "]+", flags=re.UNICODE)


class Predict(Resource):

    def connect_to_endpoint(url, headers_, params, next_token=None):
        params['next_token'] = next_token  # params object received from create_url function
        response = requests.request("GET", url, headers=headers_, params=params)
        print("Endpoint Response Code: " + str(response.status_code))
        if response.status_code != 200:
            raise Exception(response.status_code, response.text)
        return response.json()

    bearer_token = auth()
    headers_ = create_headers_(bearer_token)
    keyword = "Trump lang:en"
    max_results = 10

    url = create_url(keyword, max_results)
    json_response = connect_to_endpoint(url[0], headers_, url[1])

    with open('data.json', 'w') as f:
        json.dump(json_response, f)

    csvFile = open("data.csv", "a", newline="", encoding='utf-8')
    csvWriter = csv.writer(csvFile)

    # Create headers_ for the data you want to save, in this example, we only want save these columns in our dataset
    csvWriter.writerow(['content', 'user'])
    csvFile.close()

    append_to_csv(json_response, "data.csv")

    csvFile = open("spam.csv", "a", newline="", encoding='utf-8')
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(['content', 'user'])
    csvFile.close()

    P = pd.read_csv("data.csv")
    P['content'] = P['content'].apply(stemming)
    P = P['content'].values
    m = len(P)
    P_old = vectorizer.transform(P)
    P = transformer.transform(P_old)
    # model = copy.deepcopy(model)
    prediction = m_jlib.predict(P)
    csvFile = open("spam.csv", "a", newline="", encoding='utf-8')
    csvWriter = csv.writer(csvFile)
    P = pd.read_csv("data.csv")
    P = P['user'].values
    for i in range(m):
        if (prediction[i] == 0):
            continue
        else:
            spam = P[i]
            res = [spam]
            csvWriter.writerow(res)
            print('User', P[i], 'added to spam list\n')
    csvFile.close()


API.add_resource(Predict, '/predict')

if __name__ == '__main__':
    APP.run(debug=True, port='1080')
