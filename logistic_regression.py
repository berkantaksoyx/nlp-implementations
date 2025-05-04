import math
import datasets
import pandas as pd
import re
import string
from collections import Counter
from nltk.corpus import stopwords

data = datasets.load_dataset("imdb")

train_df = pd.DataFrame(data['train'])
test_df = pd.DataFrame(data['test'])


# First Part - Preprocessing

def preprocess_text(text):

    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

    text = re.sub(r"\d+", " ", text)

    text = text.lower()

    stop_words = set(stopwords.words("english"))
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    text = " ".join(tokens)

    text = re.sub(r"\s+", " ", text).strip()
    return text


train_df['text'] = train_df['text'].apply(preprocess_text)
test_df['text'] = test_df['text'].apply(preprocess_text)


# Second Part - Logistic Regression

def bias_scores(train_df):
    positive_reviews_merged = " ".join(train_df.loc[train_df['label'] == 1, 'text'])
    negative_reviews_merged = " ".join(train_df.loc[train_df['label'] == 0, 'text'])
    all_merged = " ".join(train_df['text'])

    vocab = set(all_merged.split())

    positive_words = positive_reviews_merged.split()
    negative_words = negative_reviews_merged.split()

    pos_counts = Counter(positive_words)
    neg_counts = Counter(negative_words)

    all_tuples = []

    for w in vocab:
        pos_freq = pos_counts[w]
        neg_freq = neg_counts[w]

        ft = pos_freq + neg_freq

        bias_value = abs(((pos_freq - neg_freq) / ft)) * math.log(ft)

        bias_tuple = (w, pos_freq, neg_freq, ft, bias_value)

        all_tuples.append(bias_tuple)

    top_10k = sorted(
        all_tuples,
        key=lambda t: (-t[4], t[0])
    )[:10000]

    return top_10k


scores = bias_scores(train_df)

print(scores[:2])
print(scores[-2:])

from sklearn.feature_extraction.text import CountVectorizer

first_10k = [t[0] for t in scores]
vectorizer = CountVectorizer(vocabulary=first_10k)

X_train = vectorizer.transform(train_df['text'])
X_test  = vectorizer.transform(test_df['text'])

y_train = train_df['label']
y_test  = test_df['label']


from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

train_accuracies = []
test_accuracies = []
iterations = list(range(1, 26))

for iteration in iterations:
    lr_model = LogisticRegression(max_iter=iteration)
    lr_model.fit(X_train, y_train)

    train_accuracies.append(lr_model.score(X_train, y_train))
    test_accuracies.append(lr_model.score(X_test,  y_test))
