# **GloVe vs fastText**  
(both are static word embeddings, but very different philosophy)

GloVe and fastText are two popular **static** word embedding methods that came after Word2Vec.  
They solve some of Word2Vec's problems, but in completely different ways.

### Quick One-line Difference

| Model    | Core Idea                                      | Best at handling...                  | Year / Creator       |
|----------|------------------------------------------------|--------------------------------------|----------------------|
| **GloVe** | Global co-occurrence statistics (whole corpus matrix) | Semantic analogies, common words     | 2014, Stanford       |
| **fastText** | Words = sum of character n-grams (subword info) | Rare words, misspellings, morphology | 2016, Facebook AI    |

### Detailed Side-by-Side Comparison Table

| Aspect                        | GloVe                                              | fastText                                                  | Winner / When to choose                     |
|-------------------------------|----------------------------------------------------|-----------------------------------------------------------|---------------------------------------------|
| **Learning signal**           | Global word–word co-occurrence counts              | Local context (like skip-gram) + subword n-grams          | —                                           |
| **How it represents a word**  | One vector per full word                           | Vector = sum of vectors of its character n-grams          | fastText (for OOV & morphology)             |
| **Out-of-vocabulary (OOV)**   | Cannot handle unseen words                         | Can generate embedding for any word (even made-up)        | **fastText wins clearly**                   |
| **Rare / infrequent words**   | Struggles (low count → noisy vector)               | Much better (inherits meaning from subwords)              | **fastText**                                |
| **Misspellings**              | Treats "cat" and "caat" as completely different    | "caat" gets almost same vector as "cat"                   | **fastText**                                |
| **Morphologically rich languages** (Turkish, Arabic, Finnish, German, Hindi…) | Average                                            | Excellent (because roots + affixes are captured)          | **fastText**                                |
| **Analogies quality** (king-man+woman≈queen) | Usually very good                                  | Good, but often slightly worse than GloVe                 | GloVe (small edge)                          |
| **Training speed**            | Slower (needs to build huge co-occurrence matrix)  | Faster (skip-gram style + hashing trick)                  | fastText                                    |
| **Model size** (on disk)      | Smaller (only word vectors)                        | Larger (n-gram vectors + words)                           | GloVe                                       |
| **Inference speed**           | Very fast (simple lookup)                          | Slightly slower (needs to sum n-grams)                    | GloVe                                       |
| **Typical dimension**         | 50–300                                             | 100–300 (often 300)                                       | —                                           |
| **Pre-trained models**        | Wikipedia, Common Crawl, Twitter, etc.             | Wikipedia, Common Crawl, many languages (~157)            | fastText (far more languages)               |
| **Still used in 2025–2026?**  | Yes — baselines, fast prototypes, when OOV not issue | Yes — especially non-English, social media, noisy text    | Both alive, fastText more in production now |

### Real-world Practical Examples

**Example 1 – Misspelling / Typo**

- Text: "I lovee this product soo much"
- GloVe: "lovee" → unknown or random vector
- fastText: "lovee" → very close to "love" because shared n-grams `<lo, lov, ove, vee, ee>`

→ fastText wins for chat, tweets, reviews, OCR output, user-generated content

**Example 2 – Rare technical word**

- Word: "neuroblastoma" (appears only few times)
- GloVe: weak / noisy vector
- fastText: strong vector because it shares subwords with "neuro", "blastoma", "neurotransmitter", "blastoma" related terms

→ fastText better for medical, legal, scientific domains

**Example 3 – Pure semantic analogy task**

- "Paris – France + Germany ≈ Berlin"
- GloVe often ranks the correct answer higher
- fastText sometimes ranks morphological neighbors higher instead

→ GloVe has slight edge on classic analogy benchmarks

### When to Choose Which (2025–2026 Practical Advice)

Use **GloVe** when:
- You work with clean, mostly English text
- You care a lot about semantic analogies / linear relationships
- You want fastest inference & smallest model size
- You're doing academic baseline comparisons

Use **fastText** when:
- You have noisy text (social media, reviews, chat, forum)
- You have many misspellings / creative spelling
- You work in morphologically rich language (most non-English languages)
- You have many rare/domain-specific words
- You need to handle completely unseen words at test time

### Quick Code – Both in gensim (very similar API)

```python
import gensim.downloader as api

# GloVe (300d Wikipedia + Gigaword)
glove = api.load("glove-wiki-gigaword-300")

# fastText (300d Wikipedia, English)
fast = api.load("fasttext-wiki-news-subwords-300")

# Same interface!
print(glove.most_similar("king"))
print(fast.most_similar("king"))

# Analogy example
print(glove.most_similar(positive=['king', 'woman'], negative=['man']))
print(fast.most_similar(positive=['king', 'woman'], negative=['man']))

# fastText can do unseen words
print(fast.most_similar("GrokAI"))          # works!
print(glove.most_similar("GrokAI"))         # KeyError / not found
```

