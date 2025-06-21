# pip install gensim==3.6.0
# pip install --upgrade scipy
# pip install gensim==3.8.3 scipy==1.5.4
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec

nltk.download('punkt')

# Sample text
text = """
Artificial intelligence and machine learning provide systems the ability to automatically learn and improve from experience without being explicitly programmed. Natural language processing is a sub-field of artificial intelligence that is focused on the interaction between computers and humans.
"""

# Tokenizing the text into sentences
sentences = sent_tokenize(text)

# Tokenizing each sentence into words
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# Get the vector for a word
word_vector = model.wv['artificial']
print("Vector for 'artificial':", word_vector)

# Find similar words
similar_words = model.wv.most_similar('artificial', topn=5)
print("Words similar to 'artificial':", similar_words)

# Save the model
model.save("word2vec_model.bin")

# Load the model
loaded_model = Word2Vec.load("word2vec_model.bin")