# Fast text 

### FastText: An In-Depth Tutorial on Word Representation

FastText is a powerful and flexible library for learning word representations and performing text classification, developed by Facebook's AI Research (FAIR) team. Unlike traditional word embedding techniques like Word2Vec, which treat each word as the smallest unit to generate vectors, FastText treats each word as composed of character n-grams. This allows FastText to generate better word embeddings for rare words, or even words not seen during training, by sharing information between similar words.

### 1. **Understanding FastText**

#### **1.1 What Makes FastText Unique?**
FastText extends the Word2Vec model by not only considering the whole word as a single unit but also breaking words down into several sub-words (n-grams). For instance, for the word "apple", and an n-gram size of 3, it would consider the subwords: "<ap", "app", "ppl", "ple", "le>", and also the whole word "<apple>" as its own token.

#### **1.2 Benefits of Using FastText**
- **Handling of Rare Words**: By breaking words into several n-grams, FastText can construct better representations for words that do not appear frequently in the corpus.
- **Word Morphology**: It is particularly effective for languages where new words can be formed by the combination of other words.
- **Robustness**: Ability to understand suffixes and prefixes.

### 2. **Installing FastText**

FastText can be used via the pre-built library in Python, which can be installed using `pip`. The library also allows users to perform operations like training supervised models and loading pre-trained word vectors.

```bash
pip install fasttext
```

### 3. **Training Word Embeddings with FastText**

#### **3.1 Preparing Your Data**
FastText expects a file with each sentence on a new line, as it does not do any tokenization itself.

```python
# Sample data
with open('data.txt', 'w') as f:
    f.write("FastText is great for text classification.\n")
    f.write("It works well with rare words.\n")
    f.write("Subword information helps with understanding morphology.\n")
```

#### **3.2 Training the Model**
You can train a FastText model using the `train_unsupervised` method which is designed to work similarly to Word2Vec.

```python
import fasttext

# Train an unsupervised model
model = fasttext.train_unsupervised('data.txt', model='skipgram', dim=100, ws=5, epoch=5)
```

#### **3.3 Saving and Loading Models**
FastText models can be saved to disk and later loaded for reuse.

```python
# Save the model
model.save_model("fasttext.model")

# Load the model
loaded_model = fasttext.load_model("fasttext.model")
```

### 4. **Using FastText Models**

#### **4.1 Accessing Word Vectors**
After training, you can get the vector for any word (even if it wasn't explicitly seen during training) due to the subword information.

```python
print(model.get_word_vector("apple"))
print(model.get_word_vector("applesauce"))  # Generates a vector even for unseen words
```

#### **4.2 Finding Similar Words**
FastText does not provide direct functionality to find similar words like Gensim's Word2Vec, but you can compute similarities using cosine similarity measures between vectors.

### 5. **Advanced Usage: Text Classification**

FastText also shines in text classification tasks, allowing for training on labeled data efficiently with the `train_supervised` function.

```python
# Example of preparing labeled data for text classification
with open('train.txt', 'w') as f:
    f.write("__label__positive I love FastText.\n")
    f.write("__label__negative FastText sometimes misses some details.\n")

# Train a supervised model
classifier = fasttext.train_supervised('train.txt')

# Predicting on new data
labels, probabilities = classifier.predict("I really like FastText!", k=1)
print(labels, probabilities)
```

### 6. **Conclusion**

FastText provides a robust, versatile approach to handling text data, particularly beneficial for languages with rich morphology or datasets with a lot of rare words. By leveraging both whole words and sub-word information, FastText creates embeddings that are rich in semantic and syntactic information, making it a powerful tool for both representation learning and text classification.
