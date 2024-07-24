# Simple NLTK ( stemming/ lemmatization / regex / stop words, corpus, unigram, bigram, trigram )
### Simple NLTK Tutorial: Core Concepts and Practical Examples

The Natural Language Toolkit (NLTK) is a powerful Python library designed for working with human language data. It includes easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning. This tutorial will cover fundamental concepts like stemming, lemmatization, regular expressions, stop words, corpora, and n-grams (unigram, bigram, trigram).

### 1. **Introduction to NLTK**

To get started with NLTK, you first need to install it and download the necessary datasets:

```bash
pip install nltk
```

In your Python environment, you can set up NLTK with:

```python
import nltk
nltk.download('popular')
```

### 2. **Text Preprocessing Techniques**

#### **2.1 Tokenization**
Breaking text into words or sentences.

```python
from nltk.tokenize import word_tokenize, sent_tokenize

text = "Hello there! How are you today? I hope you're learning a lot from this tutorial."
print(sent_tokenize(text))
print(word_tokenize(text))
```

#### **2.2 Stop Words Removal**
Removing common words that may not add much meaning to the text.

```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

words = word_tokenize(text)
filtered_words = [word for word in words if not word in stop_words]
print(filtered_words)
```

#### **2.3 Stemming**
Reducing words to their word stem or root form.

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]
print(stemmed_words)
```

#### **2.4 Lemmatization**
Similar to stemming but brings context to the words. It links words with similar meanings to one word.

```python
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
print(lemmatized_words)
```

### 3. **Working with Regular Expressions (Regex)**

Regular expressions in NLTK are used for identifying patterns in text. Here’s an example to extract all words starting with 'H' or 'h'.

```python
import re
pattern = r'\b[Hh]\w+'
matched_words = re.findall(pattern, text)
print(matched_words)
```

### 4. **Using Corpora**
NLTK provides access to many text corpora, which are large collections of text that are used to train NLP models.

```python
from nltk.corpus import gutenberg
print(gutenberg.fileids())

# Sample text from "Alice in Wonderland"
alice_text = gutenberg.raw('carroll-alice.txt')
print(alice_text[:500])  # Print first 500 characters
```

### 5. **N-Grams (Unigram, Bigram, Trigram)**

N-grams are combinations of adjacent words or letters in the text. Here’s how you can generate unigrams, bigrams, and trigrams:

```python
from nltk import bigrams, trigrams, ngrams

words = ['I', 'love', 'to', 'learn', 'NLP']
unigrams = words
bigrams_list = list(bigrams(words))
trigrams_list = list(trigrams(words))

print("Unigrams:", unigrams)
print("Bigrams:", bigrams_list)
print("Trigrams:", trigrams_list)

# General n-grams, here n=4
n_grams = list(ngrams(words, 4))
print("Four-grams:", n_grams)
```

### 6. **Conclusion**

NLTK is a comprehensive library with a vast array of functionalities for text processing and analysis. Understanding these fundamental components—stemming, lemmatization, regex, stop words, corpus usage, and n-grams—provides a solid foundation for tackling more complex NLP tasks. With these tools, you can preprocess text effectively and prepare it for deeper NLP tasks like sentiment analysis, topic modeling, or machine translation.
