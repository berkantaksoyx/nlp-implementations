import math
import datasets
import nltk
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


# Second Part - Naive Bayes
class NaiveBayesClassifier:
    def __init__(self):
        self.total_pos_words = 0
        self.total_neg_words = 0
        self.vocab_size = 0

        self.prior_pos = 0.0
        self.prior_neg = 0.0

        self.pos_counter = Counter()
        self.neg_counter = Counter()

    def fit(self, train_df):
        positive_reviews_merged = " ".join(train_df.loc[train_df['label'] == 1, 'text'])

        negative_reviews_merged = " ".join(train_df.loc[train_df['label'] == 0, 'text'])

        all_merged = " ".join(train_df['text'])

        positive_words = positive_reviews_merged.split()
        negative_words = negative_reviews_merged.split()

        all_words = all_merged.split()

        self.total_pos_words = len(positive_words)
        self.total_neg_words = len(negative_words)

        self.vocab_size = len(set(all_words))

        total_number_sample = len(train_df)
        pos_number = (train_df['label'] == 1).sum()
        neg_number = (train_df['label'] == 0).sum()

        self.prior_pos = pos_number / total_number_sample
        self.prior_neg = neg_number / total_number_sample

        self.pos_counter = Counter(positive_words)
        self.neg_counter = Counter(negative_words)

    def predict(self, text):
        text = preprocess_text(text)

        words = text.split()

        log_pos_prob = math.log(self.prior_pos)
        log_neg_prob = math.log(self.prior_neg)
        for word in words:
            pos_prob_of_word = ((self.pos_counter[word] + 1.0) / (self.total_pos_words + self.vocab_size))
            neg_prob_of_word = ((self.neg_counter[word] + 1.0) / (self.total_neg_words + self.vocab_size))

            log_pos_prob_of_word = math.log(pos_prob_of_word)
            log_neg_prob_of_word = math.log(neg_prob_of_word)

            log_pos_prob = log_pos_prob + log_pos_prob_of_word
            log_neg_prob = log_neg_prob + log_neg_prob_of_word

        y_pred = 1 if log_pos_prob > log_neg_prob else 0

        return y_pred, log_pos_prob, log_neg_prob


nb = NaiveBayesClassifier()

nb.fit(train_df)


print(nb.total_pos_words)
print(nb.total_neg_words)
print(nb.vocab_size)
print(nb.prior_pos)
print(nb.prior_neg)
print(nb.pos_counter["great"])
print(nb.neg_counter["great"])




prediction1 = nb.predict(test_df.iloc[0]["text"])
prediction2 = nb.predict("This movie will be place at 1st in my favourite movies!")
prediction3 = nb.predict("I couldn't wait for the movie to end, so I turned it off halfway through. :D It was a complete disappointment.")

print(f"{'Positive' if prediction1[0] == 1 else 'Negative'}")
print(prediction1)


print(f"{'Positive' if prediction2[0] == 1 else 'Negative'}")
print(prediction2)

print(f"{'Positive' if prediction3[0] == 1 else 'Negative'}")
print(prediction3)


print(preprocess_text("This movie will be place at 1st in my favourite movies!"))
print(preprocess_text("I couldn't wait for the movie to end, so I turned it off halfway through. :D It was a complete disappointment."))



y_true = test_df['label'].values
y_pred = [nb.predict(text)[0] for text in test_df['text']]


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")