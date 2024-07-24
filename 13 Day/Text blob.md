# Text blob

### TextBlob: Simplified Text Processing - A Comprehensive Guide

TextBlob is a Python library built on top of NLTK and Pattern, designed to provide simple APIs for common natural language processing (NLP) tasks. It is an excellent tool for beginners in NLP due to its simplicity and intuitiveness, allowing for easy and rapid text processing and analysis. This tutorial will introduce you to TextBlob, demonstrating how to perform tasks such as part-of-speech tagging, noun phrase extraction, sentiment analysis, classification, translation, and more.

### 1. **Introduction to TextBlob**

TextBlob aims to make NLP tasks accessible with a clear and concise API. Without needing to dive into the underlying algorithms, users can perform complex NLP tasks like sentiment analysis, translation, or parsing with only a few lines of code.

### 2. **Installation**

To start using TextBlob, you first need to install it along with its corpora:

```bash
pip install textblob
python -m textblob.download_corpora
```

This installation command sets up TextBlob and downloads necessary data sets for part-of-speech tagging, noun phrase extraction, and other tasks.

### 3. **Basic Usage of TextBlob**

#### **3.1 Creating a TextBlob**

First, import TextBlob and create a TextBlob object by passing a string to it:

```python
from textblob import TextBlob

text = "TextBlob is amazingly simple to use. What great fun!"
blob = TextBlob(text)
```

#### **3.2 Tokenization**

Split the text into words or sentences:

```python
print(blob.words)  # ['TextBlob', 'is', 'amazingly', 'simple', 'to', 'use']
print(blob.sentences)  # [Sentence("TextBlob is amazingly simple to use."), Sentence("What great fun!")]
```

#### **3.3 Part-of-Speech Tagging**

TextBlob can categorize words into their parts of speech:

```python
print(blob.tags)  # [('TextBlob', 'NNP'), ('is', 'VBZ'), ('amazingly', 'RB'), ('simple', 'JJ'), ('to', 'TO'), ('use', 'VB')]
```

#### **3.4 Noun Phrase Extraction**

Identify noun phrases in the text:

```python
print(blob.noun_phrases)  # ['textblob']
```

### 4. **Advanced Features**

#### **4.1 Sentiment Analysis**

TextBlob can analyze the sentiment of text using a pre-trained sentiment classifier:

```python
print(blob.sentiment)
# Sentiment(polarity=0.39166666666666666, subjectivity=0.4357142857142857)
```

This returns a `Sentiment` object with two properties, `polarity` and `subjectivity`. Polarity is a float within the range [-1.0, 1.0] where 1 means positive statement and -1 means a negative statement. Subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.

#### **4.2 Translation and Language Detection**

TextBlob simplifies translating text and detecting a text's language:

```python
# Translate to Spanish
spanish_blob = blob.translate(to='es')
print(spanish_blob)

# Detect language
print(blob.detect_language())  # 'en'
```

### 5. **Text Classification**

TextBlob supports basic classification tasks. You can train a Naive Bayes Classifier with TextBlob easily:

```python
from textblob.classifiers import NaiveBayesClassifier

train = [
  ('I love this sandwich.', 'pos'),
  ('This is an amazing place!', 'pos'),
  ('I feel very good about these beers.', 'pos'),
  ('This is my best work.', 'pos'),
  ('What an awesome view', 'pos'),
  ('I do not like this restaurant', 'neg'),
  ('I am tired of this stuff.', 'neg'),
  ('He is my sworn enemy!', 'neg'),
  ('My boss is horrible.', 'neg')
]
test = [
  ('The beer was good.', 'pos'),
  ('I do not enjoy my job', 'neg'),
  ("I ain't feeling dandy today.", 'neg'),
  ('I feel amazing!', 'pos'),
  ('Gary is a friend of mine.', 'pos'),
  ('I can't believe I'm doing this.', 'neg')
]

classifier = NaiveBayesClassifier(train)
print(classifier.classify("Their burgers are amazing"))  # 'pos'
print(classifier.accuracy(test))  # e.g., 0.83
```

### 6. **Conclusion**

TextBlob is a versatile and user-friendly library that makes it easy to get started with NLP. By leveraging TextBlob, developers can perform complex NLP tasks with straightforward and intuitive code, making it an excellent choice for prototyping and beginners looking to implement NLP concepts in real-world applications.
