# Natural Language Processing (NLP)

### 1. What is NLP?
**Natural Language Processing (NLP)** is a branch of Artificial Intelligence (AI) that helps computers understand, interpret, and generate **human language** (text or speech) in a meaningful way.

- **Natural Language Understanding (NLU)**: Machines comprehend meaning, intent, and context (e.g., "I am feeling blue" means sad, not the color).
- **Natural Language Generation (NLG)**: Machines produce human-like text (e.g., chatbots replying naturally).

**Analogy**: Imagine teaching a robot to read a book, chat with you, translate languages, or summarize news – that's NLP!

**Why is it hard?** Human language is ambiguous, full of slang, sarcasm, grammar rules, and context. "Time flies like an arrow" can mean different things!

### 2. Real-World Applications of NLP (2026 Examples)
NLP powers everyday tools:
- **Virtual Assistants**: Siri, Alexa, Google Assistant – understand voice commands.
- **Machine Translation**: Google Translate handles real-time translation.
- **Sentiment Analysis**: Companies analyze customer reviews (positive/negative).
- **Chatbots & Customer Service**: Handle queries 24/7 (e.g., Bank of America's Erica).
- **Search Engines**: Google understands intent in queries.
- **Spam Filters & Email**: Gmail sorts important vs. junk.
- **Healthcare**: Extract symptoms from doctor notes.
- **Content Creation**: Tools like Jasper generate marketing copy.
- **Voice-to-Text & Autocorrect**: On your phone.

**Lecture Tip**: Ask students – "Which app do you use daily that uses NLP?"

### 3. Core Steps in NLP Pipeline
NLP follows a pipeline: Raw text → Clean → Analyze → Model → Output.

#### Step 1: Text Preprocessing (Cleaning the Data)
Raw text is messy. We clean it first.

- **Tokenization**: Split text into words or sentences.
- **Lowercasing**: Convert to lowercase.
- **Remove Stop Words**: "the", "is", "and" (common, low-value words).
- **Stemming**: Reduce words to root (e.g., "playing" → "play").
- **Lemmatization**: Better than stemming – considers context (e.g., "better" → "good").

**Simple Example**:
Text: "The cats are playing happily in the garden!"

After preprocessing: ["cat", "play", "happy", "garden"]

#### Step 2: Basic Linguistic Tasks
- **Part-of-Speech (POS) Tagging**: Label words as noun, verb, adjective, etc.
  - Example: "The (DET) quick (ADJ) brown (ADJ) fox (NOUN) jumps (VERB)."
- **Named Entity Recognition (NER)**: Identify people, places, organizations.
  - Example: "Elon Musk works at xAI in California." → Person: Elon Musk, Org: xAI, Location: California.
- **Parsing**: Analyze sentence structure (grammar tree).

#### Step 3: Feature Representation (Turning Text into Numbers)
Machines need numbers, not words!

- **Bag of Words (BoW)**: Count word frequencies (ignores order).
- **TF-IDF**: Weighs important words (rare but meaningful).
- **Word Embeddings**: Dense vectors capturing meaning (e.g., "king" - "man" + "woman" ≈ "queen").
- **Modern**: Transformers create contextual embeddings.

### 4. NLP Techniques Evolution
1. **Rule-based** (Early): Hand-written rules (good for simple tasks, brittle).
2. **Statistical/ML** (2000s): Probabilistic models, Naive Bayes for classification.
3. **Deep Learning** (2010s+): RNNs, LSTMs for sequences.
4. **Transformers** (2017–now): Revolution! Self-attention mechanism. Models like **BERT** (Bidirectional Encoder Representations from Transformers) understand context from both sides.

**BERT Example**: Understands "bank" as river bank or financial bank based on surrounding words.

### 5. Hands-On Examples with Python (Easy for Students)
Use libraries like **NLTK** (great for learning). Install with `pip install nltk` and download data: `nltk.download('all')`.

#### Example 1: Tokenization & Preprocessing
```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

text = "Natural Language Processing is fun! The students are learning a lot."

# Tokenization
words = word_tokenize(text)
sentences = sent_tokenize(text)
print("Words:", words)

# Stop words removal
stop_words = set(stopwords.words('english'))
filtered = [w for w in words if w.lower() not in stop_words]

# Stemming & Lemmatization
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stemmed = [stemmer.stem(w) for w in filtered]
lemmatized = [lemmatizer.lemmatize(w) for w in filtered]

print("Cleaned:", filtered)
```

#### Example 2: POS Tagging & NER
```python
from nltk import pos_tag
from nltk.chunk import ne_chunk

tagged = pos_tag(words)
print("POS Tags:", tagged)

# Named Entity
tree = ne_chunk(tagged)
print(tree)  # Shows entities
```

#### Example 3: Simple Sentiment Analysis
Use VADER (in NLTK) for quick polarity scores.

**Advanced Note**: For production, use Hugging Face Transformers:
```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("I love learning NLP!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.999}]
```

### 6. Challenges in NLP
- Ambiguity (same word, different meanings).
- Sarcasm & Context.
- Low-resource languages (less data for Indian languages).
- Bias in training data.
- Computational cost (large models need GPUs).

**Solutions**: Pre-trained models (transfer learning), multilingual BERT, fine-tuning.

### 7. How to Build Your First NLP Project (Student Project Ideas)
1. **Sentiment Analyzer** for movie reviews.
2. **Chatbot** using rules or Rasa/Hugging Face.
3. **Text Summarizer**.
4. **Language Translator** clone (use APIs).
5. **Spam Detector**.

**Steps for any project**:
- Collect data (Kaggle datasets).
- Preprocess.
- Train model (scikit-learn or PyTorch).
- Evaluate (accuracy, F1-score).
- Deploy (Streamlit/Gradio).

### Summary & Key Takeaways (Easy to Remember)
- NLP = AI + Linguistics → Machines talk like humans.
- Pipeline: Preprocess → Features → Model → Insights.
- Start simple with NLTK → Advance to Transformers/BERT.
- Practice daily: Analyze WhatsApp chats or news!

-----

### Real Dataset Example (IMDB Movie Reviews Sentiment Analysis)


### Sentiment Analysis on IMDB Movie Reviews Dataset

**Why this dataset?**  
It is one of the most popular NLP benchmark datasets. It contains **50,000 real movie reviews** from IMDB, perfectly balanced:  
- 25,000 **Positive** reviews (rating ≥ 7/10)  
- 25,000 **Negative** reviews (rating ≤ 4/10)  

**Dataset Source**:  
- Original: Stanford AI (Andrew Maas et al., 2011)  
- Easy CSV version on Kaggle: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
- Columns: `review` (text) + `sentiment` (positive/negative)

**Real Sample Rows** (actual examples from the dataset style):

| Review (truncated) | Sentiment |
|--------------------|---------|
| "One of the best movies I have ever seen. The acting was superb and the story kept me hooked till the end. Highly recommended!" | Positive |
| "This movie was an absolute waste of time. Poor plot, bad acting, and it felt like it would never end. Don't watch it." | Negative |
| "A masterpiece! Brilliant direction, emotional depth, and unforgettable characters. I watched it twice in one week." | Positive |
| "Terrible script and annoying characters. I regret spending money on this. Save your time and skip it." | Negative |


### Step-by-Step Implementation

#### Step 1: Load and Explore the Dataset
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data (download CSV from Kaggle)
df = pd.read_csv('IMDB Dataset.csv')   # Shape: (50000, 2)

print(df.shape)
print(df['sentiment'].value_counts())   # Balanced: 25000 positive, 25000 negative

# Visualize
sns.countplot(x='sentiment', data=df)
plt.title('Distribution of Sentiments')
plt.show()

df.head()
```

#### Step 2: Preprocessing (Apply everything you taught earlier)
```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()                     # Lowercase
    text = re.sub(r'[^a-z\s]', '', text)    # Remove punctuation & numbers
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

# Apply to dataset (this may take 1-2 minutes)
df['clean_review'] = df['review'].apply(preprocess)

print("Original:", df['review'][0][:200])
print("Cleaned:", df['clean_review'][0][:200])
```

#### Step 3: Feature Extraction (Bag of Words / TF-IDF)
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

X = df['clean_review']
y = df['sentiment'].map({'positive': 1, 'negative': 0})   # Convert to numbers

vectorizer = TfidfVectorizer(max_features=5000)   # Limit to top 5000 words
X_tfidf = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
```

#### Step 4: Train a Simple Model (Logistic Regression – Great baseline!)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))   # Usually ~88-90% with TF-IDF
print(classification_report(y_test, y_pred))
```

**Expected Output (Typical results)**:  
Accuracy: **0.89** (89%)  
Great precision & recall for both classes.

#### Step 5: Test on New Reviews (Real-time Demo in Lecture)
```python
def predict_sentiment(review):
    clean = preprocess(review)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    return "Positive" if pred == 1 else "Negative"

# Live examples
print(predict_sentiment("The cinematography was stunning and the story moved me to tears. Best film of the year!"))
print(predict_sentiment("Boring from start to finish. Waste of 2 hours of my life."))
```

#### Step 6: Advanced – Try Transformers (Hugging Face)
For higher accuracy (~94-96%):
```python
from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

result = sentiment_pipeline("I loved this movie so much!")
print(result)   # [{'label': 'POSITIVE', 'score': 0.999}]
```

### Project Extension Ideas
1. **Improve Accuracy**: Try Naive Bayes, Random Forest, or LSTM.
2. **Word Clouds**: Show most common words in positive vs negative reviews.
3. **Error Analysis**: Find reviews where model fails (sarcasm cases).
4. **Multilingual**: Add Hindi movie reviews and compare.
5. **Deploy**: Make a Gradio web app where anyone can paste a review.

**Performance Comparison Table** (Typical student results):

| Model              | Accuracy | Training Time |
|--------------------|----------|---------------|
| TF-IDF + Logistic  | ~89%    | Fast         |
| Naive Bayes        | ~85%    | Very Fast    |
| BERT (fine-tuned)  | ~95%    | Slow (needs GPU) |

**Challenges Students Will Face**:
- Long reviews → Truncate or use padding.
- Class imbalance if they pick another dataset.
- Overfitting on common words.





