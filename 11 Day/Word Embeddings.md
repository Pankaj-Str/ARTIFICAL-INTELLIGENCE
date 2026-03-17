## **What are Word Embeddings?**

Word Embeddings = turning words into **numbers (vectors)** so that computers can understand **meaning** and **relationships** between words.

Instead of treating words as just IDs or one-hot vectors (like [0,0,1,0,...] with 1 at different position for each word), embeddings give each word a **dense vector** (e.g. 50, 100, 300 numbers) where:

- Similar meaning words → close together in this number space  
- Related concepts → follow similar directions  

**Very famous famous example everyone shows first:**

**king − man + woman ≈ queen**

This is **not magic** — it's mathematics showing the model learned:

- "king" vector – "man" vector + "woman" vector ≈ "queen" vector

This means the model understood **gender** as a direction in the vector space.

Other beautiful examples that good embeddings learn automatically:

- Paris − France + Italy ≈ Rome  
- bigger − big + small ≈ smaller  
- walking − walk + swim ≈ swimming  
- king − royal + common ≈ president (sometimes)

### Two Main Popular Methods (before modern LLMs)

| Method     | Year   | How it learns                          | Main Idea                              | Famous For                     |
|------------|--------|----------------------------------------|----------------------------------------|--------------------------------|
| Word2Vec   | 2013   | Predicts nearby words (skip-gram / CBOW) | Local context windows                  | Fast, very popular first wave  |
| **GloVe**  | 2014   | Uses **global word co-occurrence counts** | Looks at the whole corpus statistics   | Better analogies, math cleaner |

### What is GloVe? (Global Vectors for Word Representation)

- Created by Stanford researchers (2014)
- **Key idea**: Instead of looking only at nearby words (like Word2Vec), GloVe looks at **how often words appear together** in the **entire text collection** (global view)
- It builds a huge **co-occurrence matrix** → how frequently word A appears near word B across billions of words
- Then it tries to make vectors such that:

**"the dot product of two word vectors ≈ log(of how often they co-occur)"**

This simple rule surprisingly creates very good vectors with nice mathematical properties.

### Step-by-Step — How GloVe Captures Meaning (Beginner View)

Imagine we count how often words appear together in a 5-word window across Wikipedia + news + books.

Some made-up small counts:

|          | ice    | steam  | water  | fashion | dress  |
|----------|--------|--------|--------|---------|--------|
| solid    | 12     | 2      | 8      | 0       | 0      |
| gas      | 1      | 15     | 3      | 0       | 0      |
| liquid   | 6      | 4      | 20     | 0       | 1      |
| hot      | 2      | 18     | 5      | 3       | 2      |
| cold     | 14     | 1      | 7      | 1       | 0      |

GloVe sees:

- "ice" and "solid" appear together a lot → their vectors should be similar
- "steam" and "hot" appear together a lot → similar direction
- "ice" and "hot" rarely appear together → vectors far apart or opposite direction

After training → vectors learn patterns like **temperature**, **state of matter**, etc.

### Famous Real Example Everyone Remembers

**king - man + woman ≈ queen**

In numbers (real GloVe 300d values — rounded & simplified for understanding):

```
king   ≈ [2.1,  0.3,  -0.8,  ...,  gender=~2.5,  royal=~3.1]
man    ≈ [2.0,  0.4,  -0.2,  ...,  gender=~2.4,  royal=~0.1]
woman  ≈ [1.1,  0.2,  -0.1,  ...,  gender=~-2.3, royal=~0.1]
queen  ≈ [1.2,  0.3,  -0.7,  ...,  gender=~-2.4, royal=~3.0]
```

When we do **king - man** → removes "male" direction  
**+ woman** → adds "female" direction  
→ lands very close to **queen**

### Quick Python Example — Using Pre-trained GloVe (Most Practical Way)

```python
# pip install gensim  (if not already installed)

import gensim.downloader as api

# Download ~822 MB GloVe 6B tokens 300d (only once)
glove = api.load('glove-wiki-gigaword-300')   # or 'glove-twitter-25' for smaller

# Now play!
print(glove.most_similar("king"))
# [('queen', 0.7699), ('prince', 0.727...), ('emperor', ...)]

# The famous analogy
result = glove.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print(result)
# ≈ [('queen', 0.87...)]

# Another fun one
print(glove.most_similar(positive=['paris', 'italy'], negative=['france']))
# ≈ [('rome', 0.82...)]

# Distance / similarity
print("similarity(cat, dog)   =", glove.similarity('cat', 'dog'))     # ~0.80
print("similarity(cat, apple) =", glove.similarity('cat', 'apple'))   # ~0.22
```

### Summary Table — Why Beginners Should Know GloVe

| Question                     | Answer                                                                 |
|------------------------------|------------------------------------------------------------------------|
| What problem does it solve?  | Turns words → meaningful numbers (vectors)                            |
| How is GloVe different?      | Uses **global co-occurrence counts** (not just local neighbors)       |
| Famous math trick?           | king − man + woman ≈ queen                                            |
| Vector size people use?      | 50d, 100d, 200d, **300d** most common                                 |
| Where to get ready vectors?  | Stanford site, gensim, huggingface                                    |
| Still used in 2025–2026?     | Yes — in many medium-size projects, baselines, when speed matters     |

Modern LLMs (like BERT, GPT, Llama) use **contextual embeddings** (word meaning changes by sentence), but GloVe/static embeddings are still:

- Very fast
- Easy to understand
- Great for learning how vectors capture meaning

Next step suggestion:  
Run the code above → try analogies like  
`father - son + daughter`  
`biggest - big + small`  
`India - Delhi + France`
