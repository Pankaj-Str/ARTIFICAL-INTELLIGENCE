#  Artificial Neural Networks (ANNs)

### **1. Introduction to Artificial Neural Networks**

#### **1.1 What is an ANN?**
Explain that an ANN is a computational model inspired by the way neural networks in the human brain process information. It is used to approximate functions that depend on a large number of inputs that are generally unknown.

#### **1.2 History and Evolution**
Briefly touch on the history of neural networks, from the first concept in the 1940s to the development of backpropagation in the 1980s, and their resurgence with deep learning.

### 1.3 Applications of ANN

Artificial Neural Networks (ANNs) have diverse applications across various fields due to their ability to model complex patterns and relationships in data. Here are some notable applications:

#### **Image Recognition**
ANNs, especially Convolutional Neural Networks (CNNs), are fundamental in image processing and computer vision. They excel in tasks like recognizing faces, identifying objects in photos, and even analyzing satellite imagery. For instance, in facial recognition systems, ANNs learn to identify distinct features of faces, despite variations in angle, lighting, or facial expressions.

#### **Speech Recognition**
ANNs are crucial in converting spoken language into text, used in virtual assistants like Siri and Google Assistant. They analyze the sound waves and learn to recognize speech patterns and words, allowing for real-time voice-to-text conversion and understanding user commands.

#### **Medical Diagnosis**
In healthcare, ANNs are used to diagnose diseases from complex datasets derived from imaging technologies like MRIs and X-rays. For example, they can help detect tumors in medical images earlier and with more accuracy than traditional methods. Additionally, they are used in predicting disease progression and patient outcomes by learning from clinical patient data.

#### **Financial Forecasting**
ANNs predict future stock prices or market movements by learning from historical financial data. They can identify complex patterns and correlations between different market indicators that would be difficult for human analysts to find, making them invaluable tools for investment strategies and risk management.

### 2. Basics of Neural Networks

#### 2.1 Neurons - The Building Blocks

At the core of ANNs are artificial neurons, which are simplified models of biological neurons. Each neuron in an ANN functions as follows:

- **Inputs**: Each neuron receives multiple inputs from the data or from the outputs of other neurons in the previous layer.
- **Weights**: Each input is associated with a weight (a numerical value), which is adjusted during the learning process. Weights signify the importance of inputs with regard to the output.
- **Bias**: A bias term is added to the weighted sum, allowing the model to better fit the data by shifting the activation function to the left or right, which can be critical for learning patterns in data.
- **Activation Function**: The weighted sum of inputs and the bias term is then passed through an activation function, which determines whether the neuron should be activated or not. The activation function introduces non-linearity to the model, enabling it to learn complex patterns.

**Illustration of a Neuron:**
Imagine a neuron as a node with multiple lines entering it (inputs), each line having a scale (weight) that adjusts the strength of the input. The neuron processes these weighted inputs, adds a bias, and then decides how much signal to pass on to the next layer (output) based on the activation function.

Here’s a simple diagram to illustrate this:

```
 [Input1] --(Weight1)--
                        \
 [Input2] --(Weight2)---- [SUM + Bias] --(Activation Function)--> [Output]
                        /
 [Input3] --(Weight3)--
```

This setup allows the neural network to make complex decisions by combining the effects of weights, biases, and activation functions across potentially many such neurons in multiple layers, mimicking a kind of "learning" from the input data.


### 2.2 Activation Functions

Activation functions are crucial in neural networks as they introduce non-linearity to the system, allowing the network to learn complex patterns and behaviors from the data. Without non-linearity, a neural network would effectively behave like a linear regression model, limiting its ability to solve more complicated problems.

Here are some common activation functions:

#### **Sigmoid**
The sigmoid function maps any input value to an output value between 0 and 1. It's particularly useful for models where we need to predict probabilities, as the outputs can be interpreted as probabilities. It has an S-shaped curve.
![image](https://github.com/user-attachments/assets/93bfc00d-d60a-4827-8922-d44d4280e3c9)


#### **ReLU (Rectified Linear Unit)**
ReLU is a piece-wise linear function that outputs the input directly if it is positive; otherwise, it outputs zero. It is one of the most widely used activation functions in deep learning, primarily because it is simple and reduces the likelihood of the vanishing gradient problem.

![image](https://github.com/user-attachments/assets/c0604e2b-9150-4f33-bd6f-1f17e87a142e)


#### **Tanh (Hyperbolic Tangent)**
The tanh function is similar to the sigmoid but maps input values to a range between -1 and 1. It is zero-centered, making it easier in some cases for the model to learn.
- ![image](https://github.com/user-attachments/assets/6d4053bd-dd1b-4134-a6a0-1ad4ce9a4d8f)
- **Use Case**: Often used in hidden layers, particularly when data needs to be normalized around zero.

**Illustrations of Activation Functions:**
Here’s how these functions look graphically, which might help visualize how they modulate the input:

### 2.3 The Architecture of ANNs

The architecture of a typical Artificial Neural Network (ANN) is composed of three types of layers: the input layer, one or more hidden layers, and the output layer. Each layer consists of one or more neurons, as discussed earlier.

#### **Input Layer**
- **Function**: Receives the raw input data similar to the senses feeding data into the human brain.
- **Role**: Passes the data to the next layer without applying any modifications (no activation functions are used here).

#### **Hidden Layers**
- **Function**: These layers perform the bulk of the computation within the network.
- **Role**: Hidden layers are where the complex patterns in data are detected. Each hidden layer can potentially learn a different aspect of the data, with deeper layers building increasingly abstract representations. Activation functions in these layers introduce the non-linear properties needed to learn these patterns.

#### **Output Layer**
- **Function**: Produces the final output of the neural network.
- **Role**: The structure and function of the output layer vary depending on the specific task (e.g., classification, regression). For instance, in a classification task, the output layer might use a softmax activation function to distribute the outputs as a probability distribution across predicted classes.

**Illustration of ANN Architecture:**

This structure allows ANNs to tackle problems from simple linear regression to complex image recognition tasks. Each layer's output serves as the input to the next layer, creating a chain of computations that translate raw data into actionable insights.

### 3. Training Neural Networks

Training neural networks involves several critical steps, from preparing the data to adjusting the network’s internal parameters (weights and biases) through learning algorithms like backpropagation and gradient descent. Let's explore these steps with Python examples to illustrate the concepts.

#### 3.1 Data Preparation

**Importance**: Data preparation is crucial because the quality and format of your data directly influence how well your neural network can learn and perform. Key steps in data preparation include:

1. **Normalization**: This involves scaling input data to a standard range, typically 0 to 1 or -1 to 1. This is important because it ensures that no single feature dominates the others in terms of scale, leading to faster and more stable training.
   
2. **Splitting Data**: Dividing the dataset into training and testing sets helps in validating the model effectively. The training set is used to train the model, while the testing set is used to evaluate its performance to ensure that the model generalizes well to new, unseen data.

**Example with Python (Using sklearn and numpy)**:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load a simple dataset
data = load_iris()
X = data.data
y = data.target

# Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

#### 3.2 The Concept of Weights and Bias

**Overview**: Weights and biases are the learnable parameters of a neural network. During training, these parameters are adjusted to minimize the prediction error.

- **Weights** determine the influence of each input feature on the output.
- **Bias** allows the model to shift the activation function to better fit the data.

**Learning Process**: The goal is to adjust weights and biases so that the error between the predicted output and the actual output is minimized. This process involves calculating the loss (or error) and using optimization algorithms to find the optimal values.

**Example**:

Let's assume a simple model with one weight and one bias:

```python
# Initialize weight and bias
weight = np.random.normal()
bias = np.random.normal()

# Simple prediction function
def predict(x):
    return weight * x + bias

# Example of prediction
input_feature = 0.5
predicted_output = predict(input_feature)
print(f"Predicted Output: {predicted_output}")
```

#### 3.3 Backpropagation and Gradient Descent

**Backpropagation**:
- It's the process used in neural networks to minimize the loss by adjusting the weights and biases in reverse from the output back to the input layer.
- It calculates the gradient (rate of change) of the loss function with respect to each weight and bias by the chain rule of calculus.

**Gradient Descent**:
- An optimization algorithm used to minimize the loss function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient.
- In simple terms, it updates the weights and biases in the direction that decreases the loss the most.

**Python Example (Using a very simple model)**:

```python
def loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)  # Mean squared error

def update_weights(x, y, epochs, lr):
    weight = np.random.normal()
    bias = np.random.normal()

    for epoch in range(epochs):
        y_pred = x * weight + bias
        dW = -2 * np.mean(x * (y - y_pred))  # Derivative of loss w.r.t. weight
        dB = -2 * np.mean(y - y_pred)        # Derivative of loss w.r.t. bias
        
        # Update weights and biases
        weight -= lr * dW
        bias -= lr * dB
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss(y_pred, y)}")

    return weight, bias

# Example training process
x_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([2, 4, 6, 8, 10])  # Example simple linear relation y = 2x
weight, bias = update_weights(x_train, y_train, epochs=50, lr=0.01)
```

In this example, the `update_weights` function adjusts the weight and bias to minimize the loss function (`loss`) which is the mean squared error in this case. Over multiple epochs, the weight and bias are refined to approximate the underlying relationship in the training data (`y = 2x` in this example).
