# TF-IDF in NLP  


### What is TF-IDF and Why Do We Use It?
TF-IDF (Term Frequency × Inverse Document Frequency) is the most popular way to turn text into numbers for machine learning.

It answers two questions at the same time:
- How **important** is this word **in this specific document**? → Term Frequency (TF)
- How **rare** is this word **across the entire collection** of documents? → Inverse Document Frequency (IDF)

Words that appear often in one document but rarely in others get high scores (e.g., “cat” in a pet article).  
Common words like “the”, “is”, “and” get almost zero score.

Used everywhere: Google search, spam detection, recommendation systems, chatbots, etc.

---

### Step-by-Step Breakdown

#### 1. Term Frequency (TF)
**Formula**:  
TF(word, document) = (how many times word appears in the document) ÷ (total words in the document)

**Example** (Doc 1: “The cat sat on the mat” → 6 words total)  
- “cat” appears 1 time → TF = 1/6 ≈ **0.1667**  
- “the” appears 2 times → TF = 2/6 ≈ **0.3333**

#### 2. Inverse Document Frequency (IDF)
**Formula** (simple version we will use):  
IDF(word) = log( Total documents / Documents containing the word )

We use natural log (`math.log` in Python). If a word appears in every document, IDF = 0 (it is useless).

**Example** (3 documents total):
- “cat” appears in 2 documents → IDF = log(3/2) ≈ **0.4055**
- “and” appears in 1 document → IDF = log(3/1) ≈ **1.0986**
- “the” appears in all 3 documents → IDF = log(3/3) = **0**

#### 3. TF-IDF Score
**Final formula**:  
**TF-IDF** = TF × IDF

That’s it! One number per word per document.

---

### Worked Example (Super Simple Corpus)

**Our 3 documents** (corpus):
1. The cat sat on the mat  
2. The dog sat on the mat  
3. The cat and the dog  

**Unique words** (vocabulary): and, cat, dog, mat, on, sat, the

**Step-by-step manual calculation for “cat” in Doc 1**:
- TF(“cat”, Doc1) = 1 ÷ 6 ≈ 0.1667
- IDF(“cat”) = 0.4055
- TF-IDF = 0.1667 × 0.4055 ≈ **0.0676**

We repeat this for every word in every document.

**Complete TF-IDF Matrix** (rounded to 4 decimals):

|        | and    | cat    | dog    | mat    | on     | sat    | the |
|--------|--------|--------|--------|--------|--------|--------|-----|
| Doc 1  | 0.0000 | **0.0676** | 0.0000 | **0.0676** | **0.0676** | **0.0676** | 0.0 |
| Doc 2  | 0.0000 | 0.0000 | **0.0676** | **0.0676** | **0.0676** | **0.0676** | 0.0 |
| Doc 3  | **0.2197** | **0.0811** | **0.0811** | 0.0000 | 0.0000 | 0.0000 | 0.0 |

**What does this tell us?**
- In Doc 3, “and” has the highest score (0.2197) because it is rare and appears once.
- “the” is completely ignored (score 0) in all documents.
- Doc 1 and Doc 2 look similar (they share “mat”, “on”, “sat”).

---

### Python Code – From Scratch (Beginner Version)

Copy-paste and run this. No extra libraries except `pandas` (for pretty table – you can remove it if you want).

```python
import math
from collections import Counter
import pandas as pd   # optional for nice table

# ======================
# 1. Your documents
# ======================
corpus = [
    'The cat sat on the mat',
    'The dog sat on the mat',
    'The cat and the dog'
]

# Lowercase + split into words
docs = [doc.lower().split() for doc in corpus]

# All unique words
terms = sorted(set(word for doc in docs for word in doc))
print("Unique words:", terms)

N = len(docs)                     # Total documents = 3
print("Total documents:", N)

# ======================
# 2. Document Frequency (DF)
# ======================
df = {}
for term in terms:
    df[term] = sum(1 for doc in docs if term in doc)

# ======================
# 3. Inverse Document Frequency (IDF)
# ======================
idf = {term: math.log(N / df[term]) for term in terms}

# ======================
# 4. Build TF-IDF matrix
# ======================
tfidf_matrix = []
for doc in docs:
    word_counts = Counter(doc)
    total_words = len(doc)
    
    row = []
    for term in terms:
        tf = word_counts[term] / total_words          # Term Frequency
        tfidf = tf * idf[term]                        # TF × IDF
        row.append(round(tfidf, 4))
    tfidf_matrix.append(row)

# Show beautiful table
df_tfidf = pd.DataFrame(tfidf_matrix, columns=terms, 
                        index=[f'Doc {i+1}' for i in range(N)])
print("\n=== TF-IDF Matrix ===")
print(df_tfidf)
```

**Output you will see** (exactly the table above).

---

### Next Level (Optional but Useful)

If you install **scikit-learn** (most common in real projects):

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
print(X.toarray())   # TF-IDF matrix (sklearn adds smoothing + normalization)
```

The numbers will be slightly different because sklearn uses a smoothed formula, but the idea is 100% the same.

---
