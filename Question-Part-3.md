# multiple-choice questions (MCQs) with answers** 
covering the 
**Fundamentals of Artificial Neural Networks (ANN)**, including modules, activation functions, optimization functions, cost functions, dense networks, regularization, and gradient descent.  

---

### **1-10: Basics of Artificial Neural Networks (ANN)**
1. What is an Artificial Neural Network (ANN)?  
   a) A network of biological neurons  
   b) A mathematical model inspired by the human brain  
   c) A type of database  
   d) A rule-based system  
   **Answer:** b  

2. Which of the following is a key component of an ANN?  
   a) Neurons  
   b) Weights  
   c) Activation Functions  
   d) All of the above  
   **Answer:** d  

3. What is the purpose of weights in an ANN?  
   a) To store input data  
   b) To determine the strength of connections between neurons  
   c) To generate random numbers  
   d) To replace bias terms  
   **Answer:** b  

4. The basic unit of computation in an ANN is called:  
   a) A neuron  
   b) A layer  
   c) A weight  
   d) A dataset  
   **Answer:** a  

5. What are the three main layers in an ANN?  
   a) Input layer, Hidden layer(s), Output layer  
   b) Data layer, Processing layer, Decision layer  
   c) Forward layer, Backward layer, Final layer  
   d) Neuron layer, Memory layer, Output layer  
   **Answer:** a  

6. The process of adjusting weights in an ANN is called:  
   a) Forward propagation  
   b) Backpropagation  
   c) Data processing  
   d) Weight initialization  
   **Answer:** b  

7. In a neural network, the input layer:  
   a) Performs computations  
   b) Passes data to the next layer  
   c) Adjusts weights  
   d) Stores output values  
   **Answer:** b  

8. What is the primary purpose of the hidden layer(s) in an ANN?  
   a) To store input data  
   b) To extract and learn features from input data  
   c) To output final predictions  
   d) To remove redundant data  
   **Answer:** b  

9. How are the outputs of a neural network computed?  
   a) By summing the weights  
   b) By applying an activation function to weighted inputs  
   c) By averaging all inputs  
   d) By multiplying inputs with biases  
   **Answer:** b  

10. Which of the following is NOT a type of neural network?  
   a) Convolutional Neural Network (CNN)  
   b) Recurrent Neural Network (RNN)  
   c) Graph Neural Network (GNN)  
   d) Probabilistic Neural Network (PNN)  
   **Answer:** d  

---

### **11-20: Activation Functions**
11. What is the role of an activation function in a neural network?  
   a) To initialize weights  
   b) To introduce non-linearity into the network  
   c) To remove unnecessary neurons  
   d) To reduce training time  
   **Answer:** b  

12. Which activation function is commonly used in hidden layers?  
   a) Sigmoid  
   b) ReLU  
   c) Softmax  
   d) Step function  
   **Answer:** b  

13. What is a drawback of the sigmoid activation function?  
   a) Vanishing gradient problem  
   b) Exploding gradient problem  
   c) It is non-differentiable  
   d) It does not work for classification tasks  
   **Answer:** a  

14. Which activation function is most suitable for binary classification in the output layer?  
   a) Sigmoid  
   b) ReLU  
   c) Tanh  
   d) Softmax  
   **Answer:** a  

15. The ReLU activation function is defined as:  
   a) \( f(x) = 1 / (1 + e^{-x}) \)  
   b) \( f(x) = \max(0, x) \)  
   c) \( f(x) = e^x / \sum e^x \)  
   d) \( f(x) = (e^x - e^{-x}) / (e^x + e^{-x}) \)  
   **Answer:** b  

---

### **21-30: Optimization Functions & Cost Functions**
16. What is the purpose of an optimization function in an ANN?  
   a) To improve activation functions  
   b) To minimize the cost function and update weights  
   c) To increase the complexity of the model  
   d) To reduce the number of neurons  
   **Answer:** b  

17. Which of the following is NOT an optimization algorithm?  
   a) Stochastic Gradient Descent (SGD)  
   b) Adam  
   c) RMSprop  
   d) Softmax  
   **Answer:** d  

18. What is a cost function in an ANN?  
   a) A function that evaluates how well the model is performing  
   b) A function that reduces the learning rate  
   c) A function that initializes the network  
   d) A function that measures model complexity  
   **Answer:** a  

19. Mean Squared Error (MSE) is commonly used for:  
   a) Classification tasks  
   b) Regression tasks  
   c) Clustering  
   d) Dimensionality reduction  
   **Answer:** b  

20. Cross-entropy loss is best suited for:  
   a) Classification problems  
   b) Regression problems  
   c) Reinforcement learning  
   d) Clustering  
   **Answer:** a  

---

### **31-40: Dense Networks & Regularization**
21. What is a Dense Neural Network?  
   a) A network where each neuron is connected to all neurons in the next layer  
   b) A network with fewer neurons  
   c) A network with only one hidden layer  
   d) A network without activation functions  
   **Answer:** a  

22. Which of the following is a method for regularization in neural networks?  
   a) Dropout  
   b) L1 and L2 regularization  
   c) Batch Normalization  
   d) All of the above  
   **Answer:** d  

23. What is the main purpose of regularization?  
   a) To improve training speed  
   b) To prevent overfitting  
   c) To remove hidden layers  
   d) To increase model size  
   **Answer:** b  

24. L2 regularization is also known as:  
   a) Ridge regression  
   b) Lasso regression  
   c) Elastic Net  
   d) Weight decay  
   **Answer:** a  

25. Dropout regularization works by:  
   a) Adding noise to the input data  
   b) Randomly deactivating neurons during training  
   c) Reducing the learning rate  
   d) Increasing the number of hidden layers  
   **Answer:** b  

---

### **41-50: Gradient Descent**
26. What is Gradient Descent?  
   a) A method to optimize weights in an ANN  
   b) A technique to increase the learning rate  
   c) A process to initialize weights  
   d) A method for data augmentation  
   **Answer:** a  

27. Which of the following is NOT a type of Gradient Descent?  
   a) Batch Gradient Descent  
   b) Stochastic Gradient Descent (SGD)  
   c) Mini-Batch Gradient Descent  
   d) Bayesian Gradient Descent  
   **Answer:** d  

28. What is the main advantage of Stochastic Gradient Descent (SGD)?  
   a) Faster updates and convergence  
   b) Lower accuracy  
   c) It requires more data  
   d) It does not update weights  
   **Answer:** a  

29. Learning rate in Gradient Descent controls:  
   a) The size of weight updates  
   b) The number of neurons  
   c) The activation function  
   d) The number of layers  
   **Answer:** a  

30. What happens if the learning rate is too high?  
   a) The model converges quickly  
   b) The model oscillates and may not converge  
   c) The model performs better  
   d) The cost function becomes negative  
   **Answer:** b  

---

