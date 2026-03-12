## ResNet50 Example with Real Dataset (Cats vs Dogs)

![Image](https://user-images.githubusercontent.com/19996897/39514811-e8224404-4e15-11e8-9440-637536201f39.PNG)

![Image](https://camo.githubusercontent.com/2c77f234deb5e40bdfabf921a8b335690b22814c3706b196dd9c7fa0495147cb/68747470733a2f2f6d69726f2e6d656469756d2e636f6d2f6d61782f333834302f312a6f4233533579484868766f75674a6b50587563386f672e676966)

![Image](https://images.openai.com/static-rsc-3/Jp1sN5GGlAhtPlaHuJiTx6nMHx3JJhT-z0-F9qy6R1t6ozou-t6YaxnpQXBT6ppUlEYN-Rl4WnZhaU4Vl1HzmQ0bE62ZbdKFtLu8QGptvZQ?purpose=fullsize\&v=1)

![Image](https://user-images.githubusercontent.com/40482921/232365269-f5d90997-34b5-4091-9151-4df1dc13c54b.png)

This example shows how to use **ResNet-50** with a **real dataset (Cats vs Dogs)** to build an **image classification model** using **TensorFlow** and **Keras**.

We will use the **Dogs vs. Cats Dataset** which contains thousands of cat and dog images.

---

# 1. Install Required Libraries

```python
pip install tensorflow matplotlib numpy
```

---

# 2. Import Libraries

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
```

---

# 3. Load Dataset

TensorFlow provides an easy dataset loader.

```python
dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/",
    image_size=(224,224),
    batch_size=32
)
```

Folder structure should look like this:

```
dataset/
      cats/
          cat1.jpg
          cat2.jpg
      dogs/
          dog1.jpg
          dog2.jpg
```

---

# 4. Load ResNet50 Pretrained Model

We use **transfer learning** (pretrained weights from ImageNet).

```python
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)
```

Freeze pretrained layers:

```python
for layer in base_model.layers:
    layer.trainable = False
```

---

# 5. Add Custom Classification Layers

```python
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)
```

---

# 6. Compile Model

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

---

# 7. Train the Model

```python
history = model.fit(
    dataset,
    epochs=5
)
```

Example output:

```
Epoch 1/5
accuracy: 0.86
Epoch 5/5
accuracy: 0.95
```

The model learns to distinguish **cats vs dogs**.

---

# 8. Test the Model

```python
img = tf.keras.preprocessing.image.load_img(
    "test_dog.jpg",
    target_size=(224,224)
)

img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

if prediction > 0.5:
    print("Dog")
else:
    print("Cat")
```

Example result:

```
Dog
```

---

# 9. Simple Working Flow

```
Image Input
      ↓
ResNet50 Feature Extraction
      ↓
Dense Layer
      ↓
Sigmoid Output
      ↓
Prediction (Cat or Dog)
```

---

# 10. Why Use ResNet50 Here?

| Reason            | Benefit                            |
| ----------------- | ---------------------------------- |
| Pretrained Model  | Already learned millions of images |
| Deep Network      | Better feature extraction          |
| Transfer Learning | Works well with small datasets     |
| High Accuracy     | Better than simple CNN             |

---


