### Complete Example: 
# CNN for MNIST Digit Classification with Visualizations

I'll provide **two full, runnable scripts** (one in PyTorch, one in Keras/TensorFlow) that include:

- Data loading
- Model definition
- Training (5 epochs for quick demo)
- Evaluation
- Visualizations: 
  - 12 correct predictions
  - 12 misclassified examples (if available)
  - Confusion matrix

These are **step-by-step** with explanations in comments. Install required libs if needed:

- PyTorch: `pip install torch torchvision torchaudio`
- Keras/TF: `pip install tensorflow`
- Common: `pip install matplotlib scikit-learn numpy`

Both achieve ~98-99% accuracy on MNIST test set.

#### 1. PyTorch Version – Complete Code

```python
# Complete Simple CNN for MNIST with Visualizations (PyTorch)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Step 1: Load and prepare MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean & std for better training
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)

test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)  # Larger batch for faster eval

# Step 2: Define a very simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)  # Input: 1 channel (grayscale), Output: 16 features
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)  # 16 -> 32 features
        self.pool = nn.MaxPool2d(2, 2)  # Halves the spatial size
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # After two pools: 28x28 -> 14x14 -> 7x7, flatten to 32*49=1568
        self.fc2 = nn.Linear(128, 10)  # 10 output classes (digits 0-9)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # Conv + ReLU + Pool -> 14x14
        x = self.pool(self.relu(self.conv2(x)))  # Conv + ReLU + Pool -> 7x7
        x = x.view(-1, 32 * 7 * 7)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # No softmax needed (CrossEntropyLoss handles it)
        return x

model = SimpleCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 3: Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

# Step 4: Training loop (5 epochs for demo)
epochs = 5
for epoch in range(epochs):
    model.train()  # Set to training mode
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()  # Clear old gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss / len(train_loader):.4f}")

# Step 5: Evaluation and collect predictions
model.eval()  # Set to evaluation mode
correct = 0
total = 0
all_preds = []
all_labels = []
all_images = []

with torch.no_grad():
    for images, labels in test_loader:
        images_dev, labels_dev = images.to(device), labels.to(device)
        outputs = model(images_dev)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels_dev).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_images.extend(images.squeeze(1).numpy())  # Remove channel dim for plotting

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_images = np.array(all_images)

print(f"\nTest Accuracy: {100 * correct / total:.2f}%")

# Step 6: Visualization 1 - 12 Correct Predictions
correct_idx = np.where(all_preds == all_labels)[0]
np.random.seed(42)  # For reproducibility
show_idx = np.random.choice(correct_idx, size=12, replace=False)

plt.figure(figsize=(12, 8))
for i, idx in enumerate(show_idx):
    plt.subplot(3, 4, i+1)
    plt.imshow(all_images[idx], cmap='gray')
    plt.title(f"Pred: {all_preds[idx]} | True: {all_labels[idx]}")
    plt.axis('off')
plt.suptitle("12 Correct Predictions", fontsize=16)
plt.tight_layout()
plt.show()

# Step 7: Visualization 2 - 12 Misclassified Examples (if available)
wrong_idx = np.where(all_preds != all_labels)[0]
num_wrong_to_show = min(12, len(wrong_idx))
show_wrong = np.random.choice(wrong_idx, size=num_wrong_to_show, replace=False)

plt.figure(figsize=(12, 8))
for i, idx in enumerate(show_wrong):
    plt.subplot(3, 4, i+1)
    plt.imshow(all_images[idx], cmap='gray')
    plt.title(f"Pred: {all_preds[idx]} | True: {all_labels[idx]}")
    plt.axis('off')
plt.suptitle(f"{num_wrong_to_show} Misclassified Examples", fontsize=16)
plt.tight_layout()
plt.show()

# Step 8: Visualization 3 - Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))

plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix - MNIST Test Set")
plt.grid(False)
plt.show()
```

#### 2. Keras/TensorFlow Version – Complete Code

```python
# Complete Simple CNN for MNIST with Visualizations (Keras/TensorFlow)

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Step 1: Load and prepare MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to 0-1 and add channel dimension (for CNN)
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Step 2: Build a very simple CNN model
model = models.Sequential([
    layers.Conv2D(16, (5, 5), activation='relu', padding='same', input_shape=(28, 28, 1)),  # 1->16 features
    layers.MaxPooling2D((2, 2)),  # -> 14x14
    
    layers.Conv2D(32, (5, 5), activation='relu', padding='same'),  # 16->32 features
    layers.MaxPooling2D((2, 2)),  # -> 7x7
    
    layers.Flatten(),  # Flatten to 1D
    layers.Dense(128, activation='relu'),  # Fully connected
    layers.Dense(10, activation='softmax')  # 10 classes with softmax
])

model.summary()  # Optional: Print model architecture

# Step 3: Compile the model (loss, optimizer, metrics)
model.compile(optimizer='adam',  # Adam optimizer
              loss='sparse_categorical_crossentropy',  # For integer labels
              metrics=['accuracy'])

# Step 4: Train the model (5 epochs for demo)
history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=64,
                    validation_data=(x_test, y_test))  # Validate on test set during training

# Step 5: Evaluation
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

# Collect predictions
y_pred_prob = model.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = y_test

# Step 6: Visualization 1 - 12 Correct Predictions
correct_idx = np.where(y_pred == y_true)[0]
np.random.seed(42)
show_idx = np.random.choice(correct_idx, size=12, replace=False)

plt.figure(figsize=(12, 8))
for i, idx in enumerate(show_idx):
    plt.subplot(3, 4, i+1)
    plt.imshow(x_test[idx].squeeze(), cmap='gray')
    plt.title(f"Pred: {y_pred[idx]} | True: {y_true[idx]}")
    plt.axis('off')
plt.suptitle("12 Correct Predictions", fontsize=16)
plt.tight_layout()
plt.show()

# Step 7: Visualization 2 - 12 Misclassified Examples (if available)
wrong_idx = np.where(y_pred != y_true)[0]
num_wrong_to_show = min(12, len(wrong_idx))
show_wrong = np.random.choice(wrong_idx, size=num_wrong_to_show, replace=False)

plt.figure(figsize=(12, 8))
for i, idx in enumerate(show_wrong):
    plt.subplot(3, 4, i+1)
    plt.imshow(x_test[idx].squeeze(), cmap='gray')
    plt.title(f"Pred: {y_pred[idx]} | True: {y_true[idx]}")
    plt.axis('off')
plt.suptitle(f"{num_wrong_to_show} Misclassified Examples", fontsize=16)
plt.tight_layout()
plt.show()

# Step 8: Visualization 3 - Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))

plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix - MNIST Test Set")
plt.grid(False)
plt.show()
```

