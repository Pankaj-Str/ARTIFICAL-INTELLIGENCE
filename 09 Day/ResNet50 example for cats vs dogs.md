# Like the ResNet50 example for cats vs dogs

- Load your fine-tuned model  
- Drag-and-drop (or select) **your own pet photo**  
- Get a prediction: "cat" or "dog" with a confidence score

### Option 1: Quick Python script (command-line + shows the image)

Save this as `predict_my_pet.py`

```python
# predict_my_pet.py
# Run: python predict_my_pet.py

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# ─── CHANGE THIS to your saved model path ───
MODEL_PATH = "cat_dog_resnet50_finetuned.h5"   # ← update this!

if not os.path.exists(MODEL_PATH):
    print(f"Model not found at {MODEL_PATH}")
    print("Make sure you saved it after fine-tuning with: model.save('cat_dog_resnet50_finetuned.h5')")
    sys.exit(1)

# Load the model
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded.")

def predict_pet_image(image_path):
    # Load and prepare image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0   # normalize

    # Predict
    prob = model.predict(img_array)[0][0]
    label = "dog" if prob > 0.5 else "cat"
    confidence = prob if prob > 0.5 else 1 - prob
    confidence_pct = confidence * 100

    # Show result
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"{label.upper()} ({confidence_pct:.1f}% confident)")
    plt.axis('off')
    plt.show()

    print(f"\nPrediction: **{label}**  (confidence: {confidence_pct:.1f}%)")

# ─── Get image from user ───
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = input("Drag & drop your pet photo here (or type full path): ").strip().strip("'\"")

if not os.path.exists(image_path):
    print("File not found. Please check the path.")
else:
    predict_pet_image(image_path)
```

**How to use it**

1. After fine-tuning, save your model once:
   ```python
   model.save("cat_dog_resnet50_finetuned.h5")
   ```

2. Run the script:
   ```bash
   python predict_my_pet.py
   ```

3. Drag your photo into the terminal (or paste the path) → it shows the image + prediction

### Option 2: Very simple web interface (using Gradio – recommended for fun)

Install once:
```bash
pip install gradio
```

Then run this script:

```python
# pet_classifier_app.py
# Run: python pet_classifier_app.py   → opens in browser

import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

# Load your fine-tuned model
model = load_model("cat_dog_resnet50_finetuned.h5")

def classify_pet(img):
    # Prepare image
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prob = model.predict(img_array)[0][0]
    if prob > 0.5:
        label = "Dog"
        confidence = prob
    else:
        label = "Cat"
        confidence = 1 - prob

    return f"{label} ({confidence*100:.1f}% confident)"

# Create Gradio interface
demo = gr.Interface(
    fn=classify_pet,
    inputs=gr.Image(type="pil", label="Upload your pet photo 🐱🐶"),
    outputs="text",
    title="My Pet Classifier (Cat vs Dog)",
    description="Fine-tuned ResNet50 – upload a photo of your cat or dog!",
    examples=[
        ["https://images.unsplash.com/photo-1583511655857-d19b40a7a54e"],  # cute dog
        ["https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba"]   # cute cat
    ],
    cache_examples=False
)

demo.launch()
```

**What you get**
- Browser window opens automatically  
- Drag & drop photos or click to upload  
- Instant prediction with confidence %  
- Looks nice and shareable

**Quick tips**

- Works best with clear, single-pet photos (face/body visible)  
- If accuracy is low on your real photos → collect 20–50 of your own pet photos and do another round of fine-tuning  
- You can later extend it to tell apart *your* cat vs *other* cats (multi-class)

