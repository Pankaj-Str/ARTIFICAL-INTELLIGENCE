# GLOVE 


GloVe (Global Vectors for Word Representation) is technically a form of unsupervised learning because it creates embeddings by capturing global statistics of word occurrences in a text corpus without labeled data. Let’s go through an example in Python that uses pre-trained GloVe embeddings with some basic processing for unsupervised tasks, like finding word similarities.

In this example, we'll:

1. Load a pre-trained GloVe model.
2. Use it to find similarities between words and analyze the relationships between them.

### Step 1: Download Pre-trained GloVe Embeddings

Pre-trained GloVe embeddings can be downloaded from [GloVe's official website](https://nlp.stanford.edu/projects/glove/). For this example, download `glove.6B.zip`, unzip it, and select `glove.6B.100d.txt` (100-dimensional embeddings).

### Step 2: Load the GloVe Embeddings in Python

Here's a Python script to load and work with GloVe embeddings:

```python
import numpy as np

# Load GloVe embeddings
def load_glove_model(file_path):
    glove_model = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float32)
            glove_model[word] = embedding
    print(f"Loaded {len(glove_model)} words.")
    return glove_model

# Load the model
glove_model = load_glove_model("path/to/glove.6B.100d.txt")
```

### Step 3: Finding Word Similarities

Now that we have loaded the embeddings, let's create functions to calculate similarity between words and find the closest words to a given word.

#### Helper Functions for Cosine Similarity

```python
from numpy.linalg import norm

# Cosine similarity function
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

# Function to find most similar words
def find_similar_words(word, model, top_n=5):
    if word not in model:
        return f"{word} not found in GloVe model!"
    word_vec = model[word]
    similarities = {
        other_word: cosine_similarity(word_vec, other_vec)
        for other_word, other_vec in model.items()
        if other_word != word
    }
    # Sort by similarity and return the top N words
    similar_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return similar_words

# Test with an example word
similar_words = find_similar_words("king", glove_model)
print("Words similar to 'king':", similar_words)
```

### Step 4: Word Analogies (e.g., King - Man + Woman ≈ Queen)

We can also test GloVe’s ability to capture word analogies:

```python
def word_analogy(word1, word2, word3, model):
    if word1 not in model or word2 not in model or word3 not in model:
        return "One of the words not found in GloVe model!"
    
    # Calculate analogy vector: word1 - word2 + word3
    analogy_vec = model[word1] - model[word2] + model[word3]
    
    # Find the word closest to the analogy vector
    similarities = {
        other_word: cosine_similarity(analogy_vec, other_vec)
        for other_word, other_vec in model.items()
    }
    # Sort by similarity and get the closest word
    analogy_word = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[0]
    return analogy_word

# Test analogy: King - Man + Woman = ?
result = word_analogy("king", "man", "woman", glove_model)
print("Result of analogy 'king - man + woman':", result)
```

### Explanation of the Code

- **Load GloVe Model**: Loads the GloVe word vectors into a dictionary where keys are words, and values are the vectors.
- **Cosine Similarity**: A function that calculates similarity between vectors.
- **Find Similar Words**: Retrieves words closest in meaning based on cosine similarity.
- **Word Analogy**: Solves analogies by finding the word vector closest to a computed vector, representing the relationship between the given words.

This setup can be used to explore word relationships and vector operations in an unsupervised manner with GloVe embeddings.
