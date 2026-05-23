# NLTK (Natural Language Toolkit)

### 1. Installation & Setup

```bash
pip install nltk
```

```python
import nltk

# Download required resources (run once)
nltk.download(['punkt', 'stopwords', 'averaged_perceptron_tagger', 
               'wordnet', 'movie_reviews', 'brown', 'gutenberg'])
```

### 2. Key NLTK Features with Real Datasets

NLTK comes with many built-in corpora:

- **Gutenberg**: Classic literature (e.g., *Moby Dick*, Shakespeare, Jane Austen)
- **Brown**: First million-word electronic corpus (15 genres)
- **Movie Reviews**: 2000 movie reviews labeled positive/negative
- **Reuters**: News articles
- **WordNet**: Lexical database

#### Example: Loading and Exploring Corpora

```python
from nltk.corpus import gutenberg, brown, movie_reviews

# Gutenberg
print("Gutenberg files:", gutenberg.fileids()[:5])
emma = gutenberg.raw('austen-emma.txt')
print("Emma length (chars):", len(emma))
print(emma[:500])  # First 500 chars

# Brown Corpus
print("\nBrown categories:", brown.categories()[:5])
news_text = brown.words(categories='news')
print("Sample news words:", news_text[:20])

# Movie Reviews
print("\nMovie reviews categories:", movie_reviews.categories())
print("Positive files:", len(movie_reviews.fileids('pos')))
print("Negative files:", len(movie_reviews.fileids('neg')))
```

### 3. Basic Text Processing Pipeline

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string

text = """Natural Language Processing is fascinating. NLTK makes it easy to work with text!
          It supports many real datasets like movie reviews and classic books."""

# 1. Sentence Tokenization
sentences = sent_tokenize(text)
print("Sentences:", sentences)

# 2. Word Tokenization
words = word_tokenize(text)
print("Words:", words[:15])

# 3. Lowercase + Remove Punctuation + Stopwords
stop_words = set(stopwords.words('english'))
filtered_words = [word.lower() for word in words 
                  if word.lower() not in stop_words 
                  and word not in string.punctuation]

print("Filtered:", filtered_words[:15])

# 4. Stemming vs Lemmatization
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed = [stemmer.stem(word) for word in filtered_words]
lemmatized = [lemmatizer.lemmatize(word) for word in filtered_words]

print("Stemmed:", stemmed[:10])
print("Lemmatized:", lemmatized[:10])
```

### 4. Part-of-Speech (POS) Tagging

```python
from nltk import pos_tag

sample = "The quick brown fox jumps over the lazy dog."
tokens = word_tokenize(sample)
pos_tags = pos_tag(tokens)
print(pos_tags)
```

Common tags: `NN` (noun), `VB` (verb), `JJ` (adjective), etc.

### 5. Named Entity Recognition (NER)

```python
from nltk.chunk import ne_chunk

text = "Barack Obama was born in Hawaii and worked at Microsoft in 2020."
tokens = word_tokenize(text)
tagged = pos_tag(tokens)
ner_tree = ne_chunk(tagged)
print(ner_tree)
```

### 6. Real-World Example: Sentiment Analysis on Movie Reviews

This is a classic NLTK example using the built-in `movie_reviews` dataset.

```python
from nltk.corpus import movie_reviews
from nltk import FreqDist
import random

# Prepare data
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

# Feature extraction (bag of words)
all_words = [word.lower() for word in movie_reviews.words()]
word_features = list(FreqDist(all_words))[:2000]  # Top 2000 words

def find_features(document):
    words = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    return features

feature_sets = [(find_features(rev), category) for (rev, category) in documents]

# Split into train/test
train_set = feature_sets[:1600]
test_set = feature_sets[1600:]

# Train Naive Bayes Classifier
from nltk import NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(train_set)

print("Accuracy:", nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(15)
```

**Typical accuracy**: ~70-80% with this simple approach.

### 7. Frequency Distribution & Concordance

```python
from nltk import FreqDist, Text

# Using Moby Dick
moby = Text(gutenberg.words('melville-moby_dick.txt'))
print("Length of Moby Dick:", len(moby))

# Frequency
fd = FreqDist(moby)
print("Most common words:", fd.most_common(10))

# Concordance (context)
moby.concordance("whale", lines=5)
```

### 8. Collocations (Words that often appear together)

```python
print(moby.collocations())
```

### 9. Advanced: Text Classification with Custom Data

You can use any dataset (e.g., CSV of reviews, tweets, etc.) by loading it with pandas and converting to NLTK format.

### 10. Best Practices & Tips

- Always preprocess (lowercase, remove stopwords, lemmatize)
- Use `nltk.download('all')` carefully (it's large)
- For production, consider **spaCy** or **Hugging Face Transformers** (faster/more accurate)
- NLTK is great for **education** and **rule-based** systems

### Resources for Further Learning

- Official Book: *Natural Language Processing with Python* (free online)
- NLTK Documentation: https://www.nltk.org/
- Practice Datasets: Movie Reviews, Brown, Reuters

