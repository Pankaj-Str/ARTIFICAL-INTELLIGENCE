
---

## NLP in Machine Learning

### Goal: Sentiment Analysis – Detect if text is positive or negative.


---

#### Step 1: Install Required Libraries
```python

pip install scikit-learn

```
---

#### Step 2: Sample Code
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Step 1: Sample data (texts and labels)
texts = [
    "I love this movie",       # positive
    "This is an amazing book", # positive
    "I hate this place",       # negative
    "This was a terrible experience", # negative
]

labels = ["positive", "positive", "negative", "negative"]

# Step 2: Convert text to numbers (Bag of Words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Step 3: Train a model
model = MultinomialNB()
model.fit(X, labels)

# Step 4: Test on new data
test_text = ["I love this place", "It was a bad movie"]
test_data = vectorizer.transform(test_text)
predictions = model.predict(test_data)

# Step 5: Show results
for sentence, prediction in zip(test_text, predictions):
    print(f"'{sentence}' ➜ {prediction}")

```
---

Output
```python
'I love this place' ➜ positive
'It was a bad movie' ➜ negative

```
---

####  How it Works:

- CountVectorizer turns words into numbers.

- Naive Bayes learns from the words and labels.

- Predict new sentences using the learned model.



---


