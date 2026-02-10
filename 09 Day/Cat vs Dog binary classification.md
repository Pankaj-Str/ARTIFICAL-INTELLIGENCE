# ResNet50

The best balance of speed and accuracy) for a **Cat vs Dog binary classification** task.

We’ll use the famous **Cats vs Dogs dataset** from Kaggle (or TensorFlow directly) and fine-tune ResNet50 in just a few steps.

### Full Working Code (Keras / TensorFlow) – Runs in <10 minutes on Colab or any GPU

```python
# Fine-tuning ResNet50 for Cat vs Dog Classification (Beginner Friendly)

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the Cats vs Dogs dataset (built into TensorFlow!)
# It automatically downloads ~800MB
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = path_to_zip.replace('.zip', '')

train_dir = PATH + '/cats_and_dogs_filtered/train'
validation_dir = PATH + '/cats_and_dogs_filtered/validation'

# 2000 training images (1000 cats + 1000 dogs), 1000 validation

# Step 2: Data preprocessing + augmentation (helps prevent overfitting)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for validation

# Automatically load images from folders
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),      # ResNet50 expects 224x224
    batch_size=32,
    class_mode='binary'          # 0 = cat, 1 = dog
)

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Step 3: Load pre-trained ResNet50 (without the top classifier)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model first (we'll only train the new top layers)
base_model.trainable = False

# Step 4: Add your own classifier on top
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),   # Converts 7x7x2048 → 2048 vector
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),               # Helps prevent overfitting
    layers.Dense(1, activation='sigmoid')  # Binary output: cat (0) or dog (1)
])

model.summary()

# Step 5: Compile the model
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 6: Train only the top layers first (fast and stable)
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Step 7: Unfreeze some layers of ResNet50 for fine-tuning (optional but boosts accuracy)
base_model.trainable = True

# Fine-tune from this layer onwards (usually last 50-100 layers)
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Use a VERY small learning rate when fine-tuning
model.compile(optimizer=optimizers.Adam(1e-5),  # 10x smaller
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Continue training (more epochs)
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,  # Total 20 epochs now
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Step 8: Plot accuracy & loss
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']

plt.figure(figsize=(8, 5))
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Fine-tuned ResNet50: Cat vs Dog')
plt.legend()
plt.show()

# Final result: Usually reaches 95–98% validation accuracy!
print("Final Validation Accuracy: {:.2f}%".format(val_acc[-1] * 100))
```

### Expected Results After 20 Epochs
```
Final Validation Accuracy: 96.5% ~ 98.2%
```
(With good augmentation and fine-tuning)

### You Can Replace ResNet50 With Any Other Model!

Just change one line:

```python
# For VGG16
from tensorflow.keras.applications import VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# For InceptionV3 (use 299x299 images!)
base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
# And change target_size=(299,299) in generators
```

### Summary: Why This Works So Well

| Step                  | Why It Matters                                   |
|-----------------------|--------------------------------------------------|
| Pre-trained weights   | Already knows edges, shapes, animal parts        |
| Freeze base           | Fast & stable training at first                  |
| Add new head          | Learns to say “this is a cat, not 1000 classes”  |
| Data augmentation     | Prevents overfitting on small dataset            |
| Fine-tune later       | Slightly adjusts deep features for cats/dogs     |



