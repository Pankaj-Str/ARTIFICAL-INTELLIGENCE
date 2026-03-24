## Python using Word2Vec with `gensim`, which covers training a model, finding similar words, and performing word analogy tasks.

### 1. Install `gensim`

First, if you don't already have `gensim` installed, run:

```bash
pip install gensim
```

### 2. Import Libraries and Prepare the Corpus

We'll use a small corpus for demonstration, but in practice, using a large text dataset like Wikipedia or news articles will yield better embeddings.

```python
import gensim
from gensim.models import Word2Vec

# Sample corpus of sentences (tokenized)
sentences = [
    ["dog", "barks", "at", "the", "cat"],
    ["cat", "meows", "back", "at", "the", "dog"],
    ["fish", "swims", "in", "the", "pond"],
    ["bird", "flies", "over", "the", "pond"],
    ["dog", "chases", "the", "cat"],
    ["cat", "climbs", "the", "tree"],
    ["fish", "jumps", "out", "of", "the", "water"]
]
```

### 3. Train the Word2Vec Model

We’ll use the Skip-gram model by setting `sg=1`. The model parameters can be adjusted for different results.

```python
# Train Word2Vec model
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=1)

# Display the vocabulary
print("Vocabulary:", list(model.wv.index_to_key))
```

### 4. Find Similarity Between Words

Now, let's compute the similarity between two words in the vocabulary.

```python
# Word similarity
similarity = model.wv.similarity("dog", "cat")
print(f"Similarity between 'dog' and 'cat': {similarity:.4f}")
```

### 5. Find Most Similar Words

You can find the words that are most similar to a given word based on the learned embeddings.

```python
# Find similar words to 'dog'
similar_words = model.wv.most_similar("dog", topn=3)
print("Words similar to 'dog':", similar_words)
```

### 6. Perform a Word Analogy

Using Word2Vec, we can solve analogies such as `dog - barks + meows ≈ cat`.

```python
# Example analogy: 'dog' - 'barks' + 'meows'
analogy = model.wv.most_similar(positive=['dog', 'meows'], negative=['barks'], topn=1)
print("Result of analogy 'dog - barks + meows':", analogy)
```

### Full Code

Here’s the complete code with all steps included:

```python
import gensim
from gensim.models import Word2Vec

# Sample corpus of sentences (tokenized)
sentences = [
    ["dog", "barks", "at", "the", "cat"],
    ["cat", "meows", "back", "at", "the", "dog"],
    ["fish", "swims", "in", "the", "pond"],
    ["bird", "flies", "over", "the", "pond"],
    ["dog", "chases", "the", "cat"],
    ["cat", "climbs", "the", "tree"],
    ["fish", "jumps", "out", "of", "the", "water"]
]

# Train Word2Vec model with Skip-gram
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=1)

# Display the vocabulary
print("Vocabulary:", list(model.wv.index_to_key))

# Find similarity between words
similarity = model.wv.similarity("dog", "cat")
print(f"Similarity between 'dog' and 'cat': {similarity:.4f}")

# Find words similar to 'dog'
similar_words = model.wv.most_similar("dog", topn=3)
print("Words similar to 'dog':", similar_words)

# Perform analogy: 'dog' - 'barks' + 'meows'
analogy = model.wv.most_similar(positive=['dog', 'meows'], negative=['barks'], topn=1)
print("Result of analogy 'dog - barks + meows':", analogy)
```

### Expected Output

The output will depend on the training data, but might look something like this:

```plaintext
Vocabulary: ['dog', 'barks', 'at', 'the', 'cat', 'meows', 'back', 'fish', 'swims', 'in', 'pond', 'bird', 'flies', 'over', 'chases', 'climbs', 'tree', 'jumps', 'out', 'of', 'water']
Similarity between 'dog' and 'cat': 0.7892
Words similar to 'dog': [('cat', 0.7892), ('chases', 0.6457), ('barks', 0.6543)]
Result of analogy 'dog - barks + meows': [('cat', 0.8214)]
```

This example covers training, word similarity, finding similar words, and analogies using Word2Vec embeddings with a simple corpus.
