# basic NLP concepts

Imagine this sentence:  
**"The quick brown foxes are running happily in the forest yesterday."**

### 1. Corpus
- **Corpus** = A big collection of text documents.  
  It's like your "library of text" that the computer learns from.

Examples of corpus:
- All Wikipedia articles
- All tweets in English
- Harry Potter books
- Movie reviews dataset

### 2. Stop Words
- Very common words that usually **don't add much meaning** → we remove them to make text lighter.

Common stop words: the, is, are, and, in, on, at, to, a, an, of, for, with...

Before: "The quick brown foxes are running happily in the forest yesterday."  
After removing stop words: "quick brown foxes running happily forest yesterday"

### 3. Stemming
- Cuts words to their **root form** (stem) using simple rules.  
- Fast but sometimes **not perfect** (can create non-real words).

Examples:

| Original word   | After Stemming (Porter stemmer) |
|-----------------|---------------------------------|
| running         | run                             |
| runs            | run                             |
| runner          | runner                          |
| happily         | happi                           |
| foxes           | fox                             |
| yesterday       | yesterday                       |

→ "happi" is not a real word → that's why stemming is crude.

### 4. Lemmatization
- Smarter than stemming → reduces word to its **proper base form** (dictionary form / lemma).  
- Understands meaning and part-of-speech (noun/verb/etc.).

Examples:

| Original word   | After Lemmatization            |
|-----------------|--------------------------------|
| running         | run                            |
| runs            | run                            |
| runner          | runner                         |
| happily         | happy                          |
| foxes           | fox                            |
| better          | good                           |
| went            | go                             |

→ Much cleaner and real English words!

Stemming vs Lemmatization (quick summary):

Original   → Stemming → Lemmatization  
running    → run       → run  
happily    → happi     → happy  
better     → better    → good

### 5. Regex (Regular Expressions)
- Very powerful pattern-matching tool.  
- Used to find/replace/clean specific patterns in text.

Super simple examples:

| Task                              | Regex pattern     | Example what it matches              |
|-----------------------------------|-------------------|--------------------------------------|
| Find all numbers                  | \d+               | 123, 45, 2026                        |
| Remove email addresses            | [\w\.-]+@[\w\.-]+ | pankaj123@gmail.com                  |
| Find words that start with capital| ^[A-Z]            | The, Pankaj, Mumbai                  |
| Remove special characters         | [^a-zA-Z0-9\s]    | ! @ # $ % ^ & * ( )                  |

Common use: cleaning tweets → remove @usernames, #hashtags, URLs, etc.

### 6. N-grams (Unigram, Bigram, Trigram...)
- Way to break text into **small sequences of words**.

| Type      | n value | Meaning                        | Example from sentence: "I love to code" |
|-----------|---------|--------------------------------|------------------------------------------|
| **Unigram**   | 1       | Single word                    | I, love, to, code                        |
| **Bigram**    | 2       | Two words together             | I love, love to, to code                 |
| **Trigram**   | 3       | Three words together           | I love to, love to code                  |

Why use n-grams?
- Unigram → ignores order → "dog bites man" and "man bites dog" look same
- Bigram/Trigram → keeps some order & context → much better for meaning

Real-life use:
- Autocomplete (Google search suggestions) → uses bigrams & trigrams
- Spell correction
- Chatbots understand "New York" as bigram (not "new" + "york" separately)

### Quick Summary Table (Super Easy)

| Concept         | What it does                              | Example Input → Output                          |
|-----------------|-------------------------------------------|-------------------------------------------------|
| Stop words      | Remove common useless words               | the cat is → cat                                |
| Stemming        | Cut to rough root (fast, rough)           | running, runs → run                             |
| Lemmatization   | Cut to correct base word (smart)          | running, ran → run                              |
| Regex           | Find & clean patterns                     | price: $99.99 → price 99.99                     |
| Unigram         | 1 word                                    | I love NLP                                      |
| Bigram          | 2 words together                          | I love, love NLP                                |
| Trigram         | 3 words together                          | I love NLP                                      |
| Corpus          | Big text collection                       | All news articles from 2025                     |

These are the **first steps** almost everyone learns in NLP — very important for cleaning text before doing any real ML/AI work!

-----

> **"The cats are running in the garden."**

---

# 1. Stemming

**Stemming** reduces a word to its **root form** by removing suffixes like *ing, ed, ly*.

It may not always produce a real word.

### Example

| Word    | Stemmed Word |
| ------- | ------------ |
| running | run          |
| playing | play         |
| studies | studi        |
| cats    | cat          |

Example sentence after stemming:

Original

```
The cats are running in the garden
```

After stemming

```
the cat are run in the garden
```

### Python Example

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

words = ["running","playing","studies","cats"]

for word in words:
    print(stemmer.stem(word))
```

Output

```
run
play
studi
cat
```

---

# 2. Lemmatization

**Lemmatization** converts a word into its **dictionary base form (lemma)**.

It is **more accurate than stemming**.

### Example

| Word    | Lemma |
| ------- | ----- |
| running | run   |
| better  | good  |
| studies | study |
| cats    | cat   |

Example sentence after lemmatization

```
The cat be run in the garden
```

### Python Example

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

words = ["running","studies","cats"]

for word in words:
    print(lemmatizer.lemmatize(word))
```

Output

```
running
study
cat
```

---

# 3. Regex (Regular Expression)

**Regex** is used to **search, match, or clean text patterns**.

### Example

Sentence

```
My phone number is 9876543210
```

We can extract numbers using regex.

### Python Example

```python
import re

text = "My phone number is 9876543210"

numbers = re.findall(r'\d+', text)

print(numbers)
```

Output

```
['9876543210']
```

Regex is useful for:

* Email extraction
* Removing punctuation
* Finding numbers
* Cleaning text

---

# 4. Stop Words

**Stop words** are common words that **do not add important meaning**.

Examples:

```
is
the
are
in
and
a
to
```

Sentence

```
The cats are running in the garden
```

After removing stop words

```
cats running garden
```

### Python Example

```python
from nltk.corpus import stopwords

text = "The cats are running in the garden"

stop_words = set(stopwords.words("english"))

words = text.split()

filtered = [w for w in words if w.lower() not in stop_words]

print(filtered)
```

Output

```
['cats', 'running', 'garden']
```

---

# 5. Corpus

A **corpus** is a **large collection of text data** used to train NLP models.

Examples of corpus:

* Wikipedia articles
* News articles
* Books
* Tweets

Example small corpus:

```
Document 1: I love machine learning
Document 2: Machine learning is powerful
Document 3: NLP is part of AI
```

This collection of documents is called a **corpus**.

---

# 6. Unigram

A **unigram** is a **single word token**.

Sentence

```
I love NLP
```

Unigrams

```
I
love
NLP
```

### Python Example

```python
text = "I love NLP"

words = text.split()

print(words)
```

Output

```
['I','love','NLP']
```

---

# 7. Bigram

A **bigram** is a **pair of two consecutive words**.

Sentence

```
I love NLP
```

Bigrams

```
I love
love NLP
```

### Python Example

```python
from nltk.util import ngrams

text = "I love NLP"

words = text.split()

bigrams = list(ngrams(words,2))

print(bigrams)
```

Output

```
[('I','love'),('love','NLP')]
```

---

# 8. Trigram

A **trigram** is a **group of three consecutive words**.

Sentence

```
I love machine learning
```

Trigrams

```
I love machine
love machine learning
```

### Python Example

```python
from nltk.util import ngrams

text = "I love machine learning"

words = text.split()

trigrams = list(ngrams(words,3))

print(trigrams)
```

Output

```
[('I','love','machine'),('love','machine','learning')]
```

---

# Quick Summary

| Concept       | Meaning                               |
| ------------- | ------------------------------------- |
| Stemming      | Cuts word to root form                |
| Lemmatization | Converts word to dictionary base form |
| Regex         | Pattern matching in text              |
| Stop Words    | Common words removed from text        |
| Corpus        | Collection of text documents          |
| Unigram       | Single word                           |
| Bigram        | Two-word combination                  |
| Trigram       | Three-word combination                |

---


