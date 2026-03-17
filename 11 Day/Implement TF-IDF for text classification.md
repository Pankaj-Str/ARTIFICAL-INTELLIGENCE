## **two realistic ways** to implement **TF-IDF** for **text classification** in Python.

We’ll cover:

1. **Recommended way (99% of real projects)** → using `scikit-learn` (very clean & fast)
2. **Educational way** → implementing TF-IDF mostly from scratch + simple classifier

### 1. Best / Most Common Way — scikit-learn Pipeline (Recommended)

This is what you should use in almost every real project.

```python
# ────────────────────────────────────────────────
#   Text Classification with TF-IDF + Naive Bayes
# ────────────────────────────────────────────────

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ─── Sample data (sentiment classification) ───
texts = [
    "I love this movie it's amazing and beautiful",
    "This film is fantastic and super fun",
    "Best product ever highly recommend",
    "Really enjoyed the story and acting",
    "This is terrible worst movie of the year",
    "I hate this product it broke in two days",
    "Awful service never buying again",
    "Very disappointing and boring",
    "The food was disgusting and cold",
    "Don't waste your money on this garbage"
]

labels = [1,1,1,1,1, 0,0,0,0,0]   # 1 = positive, 0 = negative

# ─── Split ───
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42, stratify=labels
)

# ─── Pipeline = TF-IDF + Classifier ───
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,          # optional: limit vocabulary
        stop_words='english',
        ngram_range=(1,2),          # try unigrams + bigrams
        lowercase=True
    )),
    ('clf', MultinomialNB(alpha=0.5))   # alpha = smoothing
])

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Results
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# ─── Predict new sentences ───
new_reviews = [
    "This is the best phone I have ever used!",
    "Terrible quality, stopped working after one week."
]

predictions = pipeline.predict(new_reviews)
probs = pipeline.predict_proba(new_reviews)

for text, pred, prob in zip(new_reviews, predictions, probs):
    label = "Positive" if pred == 1 else "Negative"
    confidence = max(prob)
    print(f"{text:50} → {label} ({confidence:.3f})")
```

**Typical output style (with small data → accuracy varies):**

```
Accuracy: 1.0 (on this toy set)

              precision    recall  f1-score   support
Negative       1.00      1.00      1.00         1
Positive       1.00      1.00      1.00         2

This is the best phone I have ever used!          → Positive (0.92)
Terrible quality, stopped working after one week. → Negative (0.89)
```

### 2. Educational Version — TF-IDF from (almost) Scratch + Classifier

We implement TF-IDF manually but still use a simple classifier.

```python
import math
from collections import Counter, defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─── Documents + labels ───
docs = [
    "love amazing beautiful fantastic fun",
    "best recommend enjoyed great story",
    "terrible worst hate awful disgusting",
    "boring disappointing garbage bad service",
    "disgusting cold broke terrible never again"
]
labels = [1, 1, 0, 0, 0]

# ─── 1. Build vocabulary & document frequency ───
vocab = set()
doc_freq = Counter()

for doc in docs:
    words = doc.split()
    vocab.update(words)
    unique_words = set(words)
    for w in unique_words:
        doc_freq[w] += 1

vocab = sorted(list(vocab))
N = len(docs)

# ─── 2. IDF ───
idf = {}
for term in vocab:
    idf[term] = math.log(N / (1 + doc_freq[term]))   # smoothed version

# ─── 3. Function: document → TF-IDF vector ───
def text_to_tfidf_vector(text):
    words = text.split()
    tf = Counter(words)
    total = len(words) if len(words) > 0 else 1
    
    vector = np.zeros(len(vocab))
    for i, term in enumerate(vocab):
        if term in tf:
            tf_val = tf[term] / total
            vector[i] = tf_val * idf[term]
    return vector

# ─── 4. Convert all documents to vectors ───
X = np.array([text_to_tfidf_vector(doc) for doc in docs])
y = np.array(labels)

# ─── 5. Very simple classifier: nearest centroid ───
class NearestCentroid:
    def fit(self, X, y):
        self.centroids = {}
        for label in np.unique(y):
            mask = (y == label)
            self.centroids[label] = np.mean(X[mask], axis=0)
    
    def predict(self, X):
        preds = []
        for x in X:
            distances = {lbl: np.linalg.norm(x - cent) for lbl, cent in self.centroids.items()}
            preds.append(min(distances, key=distances.get))
        return np.array(preds)

# ─── Train / test split ───
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

clf = NearestCentroid()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy (scratch version):", accuracy_score(y_test, y_pred))

# ─── Predict new text ───
new_text = "amazing fantastic love this product"
vec = text_to_tfidf_vector(new_text)
pred = clf.predict([vec])[0]
print("Prediction:", "Positive" if pred == 1 else "Negative")
```

### Quick Comparison Table

| Approach                  | Pros                                   | Cons                                 | When to use                       |
|---------------------------|----------------------------------------|--------------------------------------|------------------------------------|
| scikit-learn Pipeline     | Fast, robust, many options, production-ready | Less educational                   | Almost always (real projects)     |
| From-scratch TF-IDF       | Deep understanding, control            | Slower, missing smoothing tricks     | Learning, interviews, courses     |

