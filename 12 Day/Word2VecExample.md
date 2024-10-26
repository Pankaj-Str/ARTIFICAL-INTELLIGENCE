### Create a Word2Vec model using a larger sample dataset, focusing on how to preprocess the text, train the model, and explore various Word2Vec functionalities.

### Example: Analyzing Text Data with Word2Vec

We’ll use a few sample paragraphs to simulate a more natural text dataset and process it with Word2Vec. 

---

### Step 1: Install and Import Libraries

If `gensim` isn’t installed, run:

```bash
pip install gensim
```

Now, import the necessary libraries:

```python
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

# Download NLTK data (only if necessary)
nltk.download('punkt')
```

---

### Step 2: Prepare the Text Data

Let’s define a sample text corpus and process it into tokenized sentences.

```python
# Sample text corpus
text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. 
Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. 
Colloquially, the term "artificial intelligence" is often used to describe machines (or computers) that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving".
"""

# Split text into sentences
sentences = sent_tokenize(text)

# Tokenize sentences into words
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# Display tokenized data
print("Tokenized Sentences:", tokenized_sentences)
```

---

### Step 3: Train the Word2Vec Model

Using the tokenized sentences, train a Word2Vec model. Adjust parameters such as `vector_size`, `window`, and `min_count` based on your needs.

```python
# Train Word2Vec model
model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, sg=1)

# Display the vocabulary
print("Vocabulary:", list(model.wv.index_to_key))
```

---

### Step 4: Find Similarity Between Words

Now that the model is trained, you can use it to calculate the similarity between words in the vocabulary.

```python
# Word similarity example
similarity = model.wv.similarity("intelligence", "machines")
print(f"Similarity between 'intelligence' and 'machines': {similarity:.4f}")
```

---

### Step 5: Find Words Most Similar to a Given Word

You can find the words that are closest in meaning to a given word, according to the model.

```python
# Find words similar to 'intelligence'
similar_words = model.wv.most_similar("intelligence", topn=3)
print("Words similar to 'intelligence':", similar_words)
```

---

### Step 6: Perform a Word Analogy

Use the model to solve word analogies. For example, you might want to find a word that completes the relationship `intelligence - machines + humans ≈ ?`.

```python
# Word analogy example: "intelligence - machines + humans"
analogy = model.wv.most_similar(positive=['intelligence', 'humans'], negative=['machines'], topn=1)
print("Result of analogy 'intelligence - machines + humans':", analogy)
```

---

### Step 7: Visualize Word Vectors (Optional)

To visualize the word vectors, we’ll use the `matplotlib` library with `TSNE` for dimensionality reduction.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Get word vectors and reduce dimensions with TSNE
words = list(model.wv.index_to_key)
word_vectors = model.wv[words]

# Reduce to 2D using TSNE
tsne = TSNE(n_components=2, random_state=42)
word_vectors_2d = tsne.fit_transform(word_vectors)

# Plot the 2D word vectors
plt.figure(figsize=(10, 6))
for i, word in enumerate(words):
    plt.scatter(word_vectors_2d[i, 0], word_vectors_2d[i, 1])
    plt.annotate(word, (word_vectors_2d[i, 0], word_vectors_2d[i, 1]))

plt.title("Word2Vec Word Embeddings Visualization")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()
```

---

### Full Code Example

Here’s the entire code, combining all steps:

```python
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Download NLTK data
nltk.download('punkt')

# Sample text corpus
text = """
Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. 
Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals. 
Colloquially, the term "artificial intelligence" is often used to describe machines (or computers) that mimic "cognitive" functions that humans associate with the human mind, such as "learning" and "problem solving".
"""

# Tokenize text
sentences = sent_tokenize(text)
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# Train Word2Vec model
model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, sg=1)

# Display vocabulary
print("Vocabulary:", list(model.wv.index_to_key))

# Word similarity
similarity = model.wv.similarity("intelligence", "machines")
print(f"Similarity between 'intelligence' and 'machines': {similarity:.4f}")

# Find similar words
similar_words = model.wv.most_similar("intelligence", topn=3)
print("Words similar to 'intelligence':", similar_words)

# Word analogy
analogy = model.wv.most_similar(positive=['intelligence', 'humans'], negative=['machines'], topn=1)
print("Result of analogy 'intelligence - machines + humans':", analogy)

# Visualization
words = list(model.wv.index_to_key)
word_vectors = model.wv[words]
tsne = TSNE(n_components=2, random_state=42)
word_vectors_2d = tsne.fit_transform(word_vectors)

plt.figure(figsize=(10, 6))
for i, word in enumerate(words):
    plt.scatter(word_vectors_2d[i, 0], word_vectors_2d[i, 1])
    plt.annotate(word, (word_vectors_2d[i, 0], word_vectors_2d[i, 1]))

plt.title("Word2Vec Word Embeddings Visualization")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()
```

---

### Explanation of the Code

- **Text Preprocessing**: The sample text is tokenized into sentences, and each sentence is further tokenized into words.
- **Model Training**: The `Word2Vec` model is trained on tokenized sentences.
- **Word Similarity and Analogy**: We use the trained model to find word similarities and analogies.
- **Visualization**: TSNE is used to reduce dimensionality, allowing visualization of word embeddings in 2D space.

This example demonstrates how to use Word2Vec in Python, from text preprocessing to word similarity, analogy, and visualization.
