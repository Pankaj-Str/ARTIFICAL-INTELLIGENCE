# CNN MNIST example — 

- We'll add three useful visualizations using **Matplotlib** and **scikit-learn**:

1. **Sample predictions** (show 12 images with predicted vs actual label)  
2. **Wrong / misclassified examples** (highlight typical mistakes the model makes)  
3. **Confusion matrix** (classic overview of where the model confuses which digits)

I'll give code snippets that work with **both PyTorch and Keras** versions from before.  
You can append these after training & evaluation.

### Preparation (common for both frameworks)

You need these imports:

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
```

Also make sure you have the test data ready:

- PyTorch: `test_loader` (batch_size can be 1000 or larger)
- Keras: `x_test`, `y_test`

### 1. Visualize Some Predictions (12 examples)

#### PyTorch version

```python
# After training — collect predictions & true labels
model.eval()
all_preds = []
all_labels = []
all_images = []   # to show later

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_images.extend(images.cpu().squeeze(1).numpy())   # remove channel dim

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_images = np.array(all_images)   # shape: (10000, 28, 28)

# Show 12 random correct predictions
correct_idx = np.where(all_preds == all_labels)[0]
np.random.seed(42)
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
```

#### Keras version (even simpler)

```python
# After model.fit(...)
y_pred_prob = model.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = y_test

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
```

### 2. Show Misclassified / Wrong Examples

Same logic — just filter wrong predictions:

```python
# PyTorch version
wrong_idx = np.where(all_preds != all_labels)[0]
if len(wrong_idx) > 12:
    show_wrong = np.random.choice(wrong_idx, size=12, replace=False)
else:
    show_wrong = wrong_idx

plt.figure(figsize=(12, 8))
for i, idx in enumerate(show_wrong):
    plt.subplot(3, 4, i+1)
    plt.imshow(all_images[idx], cmap='gray')
    plt.title(f"Pred: {all_preds[idx]} | True: {all_labels[idx]}")
    plt.axis('off')
plt.suptitle("12 Typical Mistakes (Misclassifications)", fontsize=16)
plt.tight_layout()
plt.show()
```

Keras version — replace `all_preds`/`all_labels`/`all_images` with `y_pred`/`y_true`/`x_test.squeeze(-1)`.

Common MNIST confusions you’ll see:  
4 ↔ 9, 7 ↔ 1, 3 ↔ 5/8, 2 ↔ 7, etc.

### 3. Confusion Matrix (very informative)

```python
# Works for both PyTorch and Keras (after you have y_true and y_pred)

cm = confusion_matrix(all_labels, all_preds)          # PyTorch
# or: cm = confusion_matrix(y_true, y_pred)          # Keras

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=np.arange(10))

plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix - MNIST Test Set")
plt.grid(False)
plt.show()
```

You’ll typically see a very strong diagonal (correct predictions) and only small off-diagonal values (errors).  
With a good model after 5–10 epochs → errors are usually < 1–2% per class.

### Quick Tips to Make It Even Better

- Increase epochs to 10–15 → accuracy often reaches 99.0–99.3%
- Add one Dropout layer (e.g. after the second pooling) to reduce slight overfitting
- Try batch_size=128 or 256 for faster training

