# CNN example** for **digit classification** (MNIST dataset — handwritten digits 0–9).

We'll use **two popular styles** so you can choose the one that feels easier for you:

1. **PyTorch** version (more flexible, popular in research)
2. **Keras / TensorFlow** version (cleaner for absolute beginners)

Both do the same thing: train a small CNN to recognize handwritten digits with ~98–99% accuracy in just a few minutes.

### 1. PyTorch – Simple CNN for MNIST

```python
# Very simple CNN for MNIST digit classification (PyTorch)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ─── 1. Load and prepare MNIST data ───
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))   # MNIST mean & std
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)

test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=1000, shuffle=False)

# ─── 2. Define a very simple CNN ───
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)   # 1→16 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)  # 16→32
        self.pool  = nn.MaxPool2d(2, 2)                           # halves size
        self.fc1   = nn.Linear(32 * 7 * 7, 128)                   # after two pools: 28→14→7
        self.fc2   = nn.Linear(128, 10)                           # 10 classes
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))     # → 14×14
        x = self.pool(self.relu(self.conv2(x)))     # → 7×7
        x = x.view(-1, 32 * 7 * 7)                  # flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ─── 3. Loss + Optimizer ───
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ─── 4. Training loop (very short – 5 epochs) ───
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}   Loss: {running_loss/len(train_loader):.4f}")

# ─── 5. Quick test accuracy ───
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
```

**Expected result after 5 epochs**: ~98–99% test accuracy

### 2. Keras / TensorFlow – Even Shorter Version

```python
# Very simple CNN for MNIST (Keras / TensorFlow)

import tensorflow as tf
from tensorflow.keras import layers, models

# ─── 1. Load and prepare data ───
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize & add channel dimension
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 28, 28, 1).astype("float32")  / 255.0

# ─── 2. Build very simple CNN ───
model = models.Sequential([
    layers.Conv2D(16, (5,5), activation='relu', padding='same', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(32, (5,5), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()

# ─── 3. Compile & train ───
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=64,
                    validation_data=(x_test, y_test))

# ─── 4. Final result ───
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {test_acc:.4f}  ({test_acc*100:.2f}%)")
```

**Typical output after 5 epochs**:
```
Test accuracy: 0.9870  (98.70%)
```

### Quick Summary – What You Learned From This Code

Layer               | Purpose                              | Typical size change (28×28 image)
--------------------|--------------------------------------|----------------------------------
Conv2D              | Find edges, textures, patterns       | same size (with padding)
MaxPooling2D        | Reduce size, keep strong signals     | 28→14, 14→7
Flatten             | Prepare for dense layers             | 7×7×32 → 1568 numbers
Dense               | Combine everything → decision        | → 10 classes

Both codes are **minimal** yet already strong enough to get very good results on MNIST.

