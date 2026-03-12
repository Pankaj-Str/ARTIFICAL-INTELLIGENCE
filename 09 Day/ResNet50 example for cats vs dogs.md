## ResNet50 (Residual Network)

![Image](https://images.openai.com/static-rsc-3/XmrmaflMqFcumrDWaK_VcNyg_sknQTDu8bqpyhht048ATYt1nn78UVRgnjR-xnkQI36N02LJXf4QTZu7AvZ9AIOe7o751wWcfWgWnKljegY?purpose=fullsize\&v=1)

![Image](https://images.surferseo.art/628dd72f-587a-429b-84c6-96128f3a47fd.png)

### 1. What is ResNet50?

**ResNet50** is a deep learning model used for **image classification**.
It was introduced by researchers from Microsoft Research in the ImageNet Large Scale Visual Recognition Challenge.

* **ResNet = Residual Network**
* **50 = number of layers**

ResNet solves the **vanishing gradient problem** in very deep neural networks by using **skip connections (residual connections)**.

**Simple idea:**

Instead of learning:

[
H(x)
]

ResNet learns the **residual**:

[
F(x) = H(x) - x
]

Then output becomes:

[
Output = F(x) + x
]

This allows the network to train **very deep models (50+ layers)** efficiently.

---

## 2. Where ResNet50 is Used

ResNet50 is widely used in AI applications such as:

1. **Image Classification**

   * Detecting objects in images

2. **Medical Imaging**

   * Tumor detection from MRI/CT scans

3. **Face Recognition**

4. **Self-Driving Cars**

   * Object detection (cars, pedestrians)

5. **Transfer Learning**

   * Pretrained model for new datasets

---

# 3. ResNet50 Example Using Python

We will use **TensorFlow + Keras** to classify an image.

### Install Libraries

```python
pip install tensorflow numpy matplotlib
```

---

### Python Example (Image Classification)

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load pretrained ResNet50 model
model = ResNet50(weights='imagenet')

# Load image
img_path = "dog.jpg"
img = image.load_img(img_path, target_size=(224,224))

# Convert image to array
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Preprocess image
x = preprocess_input(x)

# Prediction
preds = model.predict(x)

# Show top predictions
print("Predicted:", decode_predictions(preds, top=3)[0])
```

---

## 4. Example Output

```
Predicted:
1. Labrador retriever – 92%
2. Golden retriever – 5%
3. Tennis ball – 1%
```

This means the model thinks the image most likely contains a **Labrador dog**.

---

# 5. Why ResNet50 is Powerful

| Feature           | Explanation           |
| ----------------- | --------------------- |
| Deep Network      | 50 layers             |
| Skip Connections  | Helps gradients flow  |
| Pretrained Models | Trained on ImageNet   |
| High Accuracy     | Excellent performance |

---

# 6. Simple Real-World Example

Imagine you want to build:

**AI system to detect fruits**

Input image → 🍎🍌🍊
ResNet50 analyzes the image and outputs:

```
Apple – 95%
Banana – 3%
Orange – 2%
```

This is how **image classification AI systems work**.

---

**Summary**

* **ResNet50** is a deep convolutional neural network.
* Uses **skip connections** to train deep models.
* Commonly used for **image classification and transfer learning**.
* Easy to use with **TensorFlow / Keras pretrained models**.

---

