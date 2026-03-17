## **Word2Vec**  


Word2Vec (2013, Tomas Mikolov + Google team) is **not a single model** — it is a family of **two architectures** + **two training tricks** that made high-quality static word embeddings practical on huge datasets for the first time.

### Core Idea in One Sentence

**A word is characterized by the company it keeps.**  
→ Words that appear in similar contexts should have similar vector representations.

This idea is very old (distributional hypothesis — Harris & Firth 1950s), but Word2Vec made it **computationally realistic** on billions of words.

### The Two Main Architectures

| Model       | Input                        | Output to predict               | Best for                              | Speed vs Quality trade-off          |
|-------------|------------------------------|----------------------------------|----------------------------------------|--------------------------------------|
| **CBOW**    | context words (bag)          | → target / center word           | Syntactic tasks, smaller datasets      | **Faster** to train                  |
| **Skip-gram**| center / target word        | → context words (multiple)       | Semantic analogies, rare words         | **Slower** but usually **better quality** |

**Most used combination in practice (2013–2018 era)**:  
**Skip-gram + Negative Sampling**  
(very often just called "Skip-gram model" in papers & libraries)

#### 1. CBOW – Continuous Bag-of-Words

**Intuition**  
"You have several context words → guess the missing word"

Example sentence:  
"The cat sat on the mat"

Window size = 2 → contexts around "sat":

- context = [The, cat, on, the]  
- target   = sat

Model tries to predict **sat** from average of context vectors.

Architecture (shallow neural net):

```
Input layer    → one-hot × V (very sparse)
               ↓
Average        → context vector (sum or average of context word vectors)
               ↓
Hidden layer   → projection (this is the embedding matrix W we want!)
               ↓
Output layer   → softmax over V words → probability of each possible target word
```

**Advantage**: smooths noise, faster because it averages contexts  
**Disadvantage**: less sensitive to rare words & order

#### 2. Skip-gram (the famous one)

**Intuition**  
"Given this word → predict words that appear around it"

Same sentence, same window:

- target = sat  
- context words to predict = The, cat, on, the

Model tries to predict **four** words from **one** input word.

Architecture:

```
Input layer    → one-hot for target word
               ↓
Hidden layer   → embedding lookup (W matrix → dense vector we want)
               ↓
Output layer   → softmax over V words (but repeated for each context position)
```

Because we predict multiple context words → learns better representations, especially for infrequent words.

### The Big Problem – Why Full Softmax is Impossible

Vocabulary V = 100,000 – 3,000,000 words  
For each training example we do:

- Forward pass: softmax over **millions** of classes  
→ O(V) time per example → impossible on billions of tokens

Two famous solutions (both introduced / popularized by Word2Vec paper):

#### A. Hierarchical Softmax (2005 idea, used in Word2Vec)

Idea: instead of softmax over V words → build **binary tree** (Huffman tree — frequent words closer to root)

- Each leaf = one word
- Probability = product of probabilities along path from root to leaf
- Only log₂(V) ≈ 17–22 sigmoid evaluations instead of V

→ Training complexity: **O(log V)** per update

Still used sometimes when you want exact probabilities.

#### B. Negative Sampling (the one almost everyone uses)

**Key insight**:  
We don't need good probabilities for **all** words — we just need the **true context word to have higher score** than most random words.

So instead of computing full softmax:

1. Take the **real** (target, context) pair → should get high score
2. Sample **k negative** (fake) words that do **not** appear in this context (usually k=5–20)
3. Train a **binary logistic regression** classifier:
   - Positive example: real context word → label = 1
   - Negative examples: random words → label = 0

Loss becomes:

```
Loss = -log σ(u_oᵀ v_c)  −  ∑_{i=1}^k log σ( - u_{w_i}ᵀ v_c )
```

- v_c = center word embedding (input)
- u_o = output embedding of real context word
- u_{w_i} = output embeddings of negative samples
- σ = sigmoid

→ Only **k+1** dot products instead of V → **much faster**

**Very important**: Negative Sampling **does not produce a probability distribution** — it only pushes real pairs up and random pairs down.

### Quick Summary Table – Word2Vec Variants

| Variant                        | Speed       | Quality     | Use case                              | Still common in 2025–2026? |
|--------------------------------|-------------|-------------|----------------------------------------|-----------------------------|
| Skip-gram + Negative Sampling  | Very fast   | Excellent   | Most projects 2013–2018                | Yes (baselines, small models) |
| Skip-gram + Hierarchical Softmax | Medium     | Good        | When you need actual probabilities     | Rarely                      |
| CBOW + Negative Sampling       | Fastest     | Good        | Fast training, syntax-heavy tasks      | Sometimes                   |
| CBOW + Hierarchical Softmax    | Fast        | Medium      | Rarely used                            | Almost never                |

### Famous Results Everyone Remembers

After training on ~100 billion words Google News corpus (300 dim):

- king − man + woman ≈ queen (cosine similarity ~0.8)
- Paris − France + Italy ≈ Rome
- bigger − big + small ≈ smaller
- walking − walk + swim ≈ swimming

These analogies work surprisingly well because many linguistic relationships become **linear** in the vector space.

### Practical Code – gensim (most popular way)

```python
import gensim.downloader as api

# ~1.6 GB download – only once
model = api.load("word2vec-google-news-300")

# Famous analogy
print(model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1))
# → ≈ [('queen', 0.7118)]

# Similarity
print(model.similarity('cat', 'dog'))      # ~0.80
print(model.similarity('cat', 'car'))      # ~0.28

# Most similar words
print(model.most_similar("computer", topn=5))
```

### Quick Comparison – Word2Vec vs GloVe (your previous question)

| Aspect               | Word2Vec                          | GloVe                                 |
|----------------------|------------------------------------|---------------------------------------|
| Learning signal      | Local context windows             | Global co-occurrence matrix           |
| Training objective   | Predict context (or target)       | Least squares on log co-occurrences   |
| Rare words           | Skip-gram better                  | Slightly worse                        |
| Analogies            | Very good                         | Often slightly better                 |
| Speed (2013 era)     | Very fast with neg. sampling      | Slower (matrix factorization)         |

Both are **static** embeddings → same vector for "bank" whether river or finance.

Modern transformers give **contextual** embeddings → different vector depending on sentence.

But **Word2Vec still taught in every serious NLP course in 2026** because:

- Extremely educational
- Shows how meaning can emerge from simple statistics
- Very fast & lightweight
- Baselines are still useful

