# basic multiple-choice questions (MCQs) with answers
covering **Natural Language Processing (NLP)** topics, including NLP models, NLTK, stemming, lemmatization, regular expressions, stop words, corpus, and n-grams.  

---

### **1-10: Basic Introduction to NLP Models**
1. What does NLP stand for?  
   a) Neural Language Processing  
   b) Natural Learning Processing  
   c) Natural Language Processing  
   d) Non-Linear Processing  
   **Answer:** c  

2. Which of the following is NOT an application of NLP?  
   a) Machine Translation (Google Translate)  
   b) Sentiment Analysis  
   c) Image Recognition  
   d) Speech Recognition  
   **Answer:** c  

3. Which of the following is an example of an NLP task?  
   a) Speech-to-Text Conversion  
   b) Chatbot Responses  
   c) Text Summarization  
   d) All of the above  
   **Answer:** d  

4. What is the main goal of NLP?  
   a) To analyze and generate human language  
   b) To translate programming languages  
   c) To store large datasets  
   d) To design artificial neural networks  
   **Answer:** a  

5. Which of the following libraries is widely used for NLP in Python?  
   a) NumPy  
   b) TensorFlow  
   c) NLTK  
   d) Matplotlib  
   **Answer:** c  

6. What is the role of a tokenizer in NLP?  
   a) To count words  
   b) To split text into smaller components like words or sentences  
   c) To convert text into numbers  
   d) To translate text  
   **Answer:** b  

7. Which type of NLP model converts text into numerical representations?  
   a) Word Embeddings  
   b) Decision Trees  
   c) Support Vector Machines  
   d) K-Means Clustering  
   **Answer:** a  

8. Named Entity Recognition (NER) is used to identify:  
   a) Only numbers in a text  
   b) Important entities like names, locations, and dates  
   c) Grammar mistakes  
   d) The length of a sentence  
   **Answer:** b  

9. Which NLP model is used for predicting the next word in a sentence?  
   a) Decision Tree  
   b) Recurrent Neural Network (RNN)  
   c) Convolutional Neural Network (CNN)  
   d) Naïve Bayes  
   **Answer:** b  

10. What is the main challenge in NLP?  
   a) Understanding the context and meaning of words  
   b) Running faster algorithms  
   c) Increasing data storage  
   d) Improving hardware efficiency  
   **Answer:** a  

---

### **11-20: Simple NLTK**
11. What is NLTK?  
   a) A machine learning algorithm  
   b) A Python library for NLP tasks  
   c) A type of deep learning model  
   d) A programming language  
   **Answer:** b  

12. Which function in NLTK is used for tokenization?  
   a) nltk.tokenize()  
   b) nltk.split()  
   c) nltk.parse()  
   d) nltk.divide()  
   **Answer:** a  

13. In NLTK, a **Corpus** is:  
   a) A single word  
   b) A collection of text documents  
   c) A type of algorithm  
   d) A stop word  
   **Answer:** b  

14. Which of the following is an example of a stop word?  
   a) Machine  
   b) Learning  
   c) The  
   d) Python  
   **Answer:** c  

15. What is the purpose of stop words removal?  
   a) To increase sentence length  
   b) To remove common words that do not add meaning  
   c) To create longer documents  
   d) To replace missing words  
   **Answer:** b  

16. Which of the following methods is used to filter stop words in NLTK?  
   a) stopwords.words()  
   b) remove_stopwords()  
   c) clean_words()  
   d) stopword.filter()  
   **Answer:** a  

17. What is the purpose of the NLTK `pos_tag()` function?  
   a) To find synonyms  
   b) To determine parts of speech (POS)  
   c) To count the number of words  
   d) To translate text  
   **Answer:** b  

18. What does POS tagging stand for?  
   a) Pre-Ordered Syntax  
   b) Part-Of-Speech Tagging  
   c) Predictive Output Structure  
   d) Process-Oriented Sentences  
   **Answer:** b  

19. Which dataset in NLTK contains a large collection of words and their meanings?  
   a) WordNet  
   b) NLTK Corpus  
   c) StopWords  
   d) Named Entity Recognition (NER)  
   **Answer:** a  

20. The process of breaking a sentence into words is called:  
   a) Tokenization  
   b) Parsing  
   c) Translation  
   d) Speech Recognition  
   **Answer:** a  

---

### **21-30: Stemming & Lemmatization**
21. What is stemming in NLP?  
   a) Removing stop words  
   b) Converting words to their base/root form  
   c) Translating text  
   d) Finding synonyms  
   **Answer:** b  

22. Which of the following is a stemming algorithm?  
   a) Porter Stemmer  
   b) Word2Vec  
   c) TF-IDF  
   d) Softmax  
   **Answer:** a  

23. What is the output of Porter Stemmer for the word **"running"**?  
   a) Running  
   b) Run  
   c) Runs  
   d) Runner  
   **Answer:** b  

24. How is Lemmatization different from Stemming?  
   a) Lemmatization provides a meaningful root word  
   b) Stemming always gives correct words  
   c) Lemmatization does not use dictionaries  
   d) Stemming is more accurate  
   **Answer:** a  

25. Which function is used for lemmatization in NLTK?  
   a) WordNetLemmatizer()  
   b) PorterStemmer()  
   c) Tokenizer()  
   d) StopWords()  
   **Answer:** a  

---

### **31-40: Regular Expressions & Stop Words**
26. What are regular expressions used for in NLP?  
   a) Text matching and pattern recognition  
   b) Machine translation  
   c) Grammar correction  
   d) Sentiment analysis  
   **Answer:** a  

27. Which of the following is a regex pattern for finding digits?  
   a) \d+  
   b) \w+  
   c) \s+  
   d) \t+  
   **Answer:** a  

28. Which of the following words is usually considered a stop word?  
   a) Python  
   b) The  
   c) Machine  
   d) Algorithm  
   **Answer:** b  

29. Why are stop words removed in NLP?  
   a) They do not contribute much to meaning  
   b) They improve text readability  
   c) They reduce processing time  
   d) All of the above  
   **Answer:** d  

30. What is a corpus in NLP?  
   a) A single document  
   b) A collection of texts  
   c) A synonym dictionary  
   d) A grammar-checking tool  
   **Answer:** b  

---

### **41-50: N-Grams (Unigram, Bigram, Trigram)**
31. What is a unigram?  
   a) A single word in a sequence  
   b) A pair of words  
   c) Three words together  
   d) A sentence  
   **Answer:** a  

32. What is a bigram?  
   a) A single word  
   b) A sequence of two words  
   c) Three words together  
   d) A type of stemming  
   **Answer:** b  

33. What is a trigram?  
   a) Three consecutive words  
   b) A sentence  
   c) A single character  
   d) A machine learning model  
   **Answer:** a  

---

