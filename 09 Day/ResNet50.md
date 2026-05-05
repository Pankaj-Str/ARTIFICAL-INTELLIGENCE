## 1. Introduction to ResNet

### What is a Neural Network?

A **neural network** is a computational model inspired by how the human brain processes information. It is made up of layers of interconnected units called neurons.

* **Input layer**: receives raw data (for example, an image).
* **Hidden layers**: process the data through weights and activation functions.
* **Output layer**: produces the final prediction (for example, “cat” or “dog”).

Each connection has a weight, and during training the network learns the best weights to minimize error.

In simple terms:
A neural network learns patterns from data by adjusting numbers (weights) so it can make accurate predictions.

---

### Problem with Deep Neural Networks

As neural networks became deeper (more layers), researchers expected better performance. In theory, deeper networks can learn more complex patterns.

However, in practice, two major problems appeared:

1. **Vanishing Gradient Problem**
2. **Degradation Problem**

These issues made very deep networks difficult to train.

---

### Vanishing Gradient Problem

To understand this, we need to look at how neural networks learn.

* Training uses **backpropagation**, where errors are passed backward from output to input.
* Gradients (small numerical values) tell the model how much to update weights.

In deep networks:

* As gradients move backward through many layers, they become very small.
* Eventually, they become almost zero.

This causes:

* Early layers stop learning.
* Training becomes extremely slow or completely stuck.

Example idea:
If you multiply many small numbers:
0.9 × 0.9 × 0.9 × ... → becomes very close to 0

That is exactly what happens with gradients in deep networks.

---

### Degradation Problem

This is different from vanishing gradients.

Observation:

* A deeper network should perform better than a shallow one.
* But in reality, after a certain depth, accuracy starts getting worse.

This is called the **degradation problem**.

Important point:

* This is not due to overfitting.
* Even training accuracy becomes worse, not just validation accuracy.

Why it happens:

* It becomes harder for deeper networks to learn the correct mapping.
* Ideally, deeper layers should at least copy the previous layers (identity mapping), but they fail to do so.

---

### Why ResNet Was Introduced

To solve these problems, researchers from Microsoft Research introduced **ResNet (Residual Network)**.

Main goal:

* Make very deep networks easier to train.
* Solve vanishing gradient and degradation problems.

Key idea:
Instead of forcing layers to learn a direct mapping:

H(x)

ResNet lets layers learn a **residual function**:

F(x) = H(x) − x

So the final output becomes:

H(x) = F(x) + x

This is implemented using **skip connections (shortcut connections)**.

---

### Overview of ResNet

ResNet is a deep neural network architecture that introduces **skip connections**.

What makes ResNet different:

* It allows information to bypass some layers.
* It makes gradient flow easier during backpropagation.
* It enables training of very deep networks (50, 101, 152 layers).

Basic idea of a residual block:

* Input goes through some layers.
* The original input is added back to the output.

So instead of learning everything from scratch, the network learns only the **difference (residual)**.

Benefits:

* Faster training
* Better accuracy
* Stable performance even with very deep networks

---

### Simple Intuition

Think of it like this:

Instead of learning a completely new function:
“Learn full transformation”

ResNet says:
“Just learn what is different from the input”

This makes learning much easier and more efficient.

---

## What is ResNet50?

**ResNet50** is a deep convolutional neural network (CNN) used mainly for image-related tasks. It belongs to the ResNet (Residual Network) family, which introduced a smarter way to train very deep networks.

In simple terms:
ResNet50 is a powerful model that can look at an image and understand what is inside it—such as identifying objects, faces, or patterns.

It became very popular because it solved major problems that earlier deep networks faced and achieved high accuracy on large datasets like ImageNet.

---

## Meaning of “50” (50-layer deep network)

The “50” in ResNet50 refers to the **number of layers** in the network.

* It has **50 learnable layers** (mainly convolutional layers and a final fully connected layer).
* These layers are stacked in a structured way using **residual blocks**.

Why deeper networks matter:

* More layers = ability to learn more complex features
* Early layers learn simple features (edges, colors)
* Middle layers learn shapes and textures
* Deeper layers learn objects (faces, cars, animals)

Problem before ResNet:

* Increasing layers beyond a limit made performance worse

Solution in ResNet50:

* Uses **skip connections**, allowing deep networks to work effectively

---

## Developed by Microsoft Research

ResNet was introduced by Microsoft Research in 2015.

Key contributors:

* Kaiming He
* Xiangyu Zhang
* Shaoqing Ren
* Jian Sun

Achievement:

* Won the **ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2015**
* Achieved very low error rates compared to previous models

Impact:

* Changed how deep neural networks are designed
* Became a standard model in computer vision

---

## Key Idea: Residual Learning

The core concept behind ResNet50 is **Residual Learning**.

### Traditional Learning

A normal neural network tries to learn:
H(x)

This means:

* Input → directly transformed into output

### Residual Learning

ResNet changes the approach:

Instead of learning H(x), it learns:

F(x) = H(x) − x

So the final output becomes:

H(x) = F(x) + x

### What does this mean?

* The model learns only the **difference (residual)** between input and output
* It adds the original input back using a **skip connection**

### Why this helps

1. Easier learning
   Learning small changes is easier than learning full transformation

2. Better gradient flow
   Gradients can flow directly through skip connections

3. Avoids degradation
   Even very deep networks perform well

### Simple intuition

If you already have a rough answer, it is easier to correct mistakes than to solve everything from scratch.

That is exactly what ResNet50 does.

---

## Real-World Applications of ResNet50

ResNet50 is widely used in real-world AI systems, especially in computer vision.

---

### 1. Image Classification

![Image](https://images.openai.com/static-rsc-4/IBxkJoh8VBXCEX4yMpfpXvoZdFdEwaPx0ImFfBpaorL6J62Gn-WHfVaRsiqLVeb0Kv0gRyH32dyzeV1ZTXzyzd8upihG-WXm9XeJgcsjRWvZI3Fe7pXwWrUF_4d-lQcmBIZagWKE8lE7pyUpZh2gfnGbJWu4XJb-MPWnA6e9WXCLwQU_v6g-uORDow4N3HeC?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/MIBknjZRbmscRm9z62P8es_La1PGtYOGz8_ixewFekckeGkq1DzzKagf4uX9KpWsF0DXxue3o3wy4uAvr-FsPl1GsJjKtgrszdmoKOOkty3LlTIPlQrih55G6iFqZlL4xnzEkYLfXfIifKJ7ik0XIQ_8cfpOBOAFM0Y5hRhE6ByphCC-XfqqdiR8hRn5E5jd?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/yAp_uuGDOwbmcZnV6GpXr_eMZ3XD2Y_1BE6cX_jZVRKODDPMPtG3ORHy9zEZQ0Ik9RTwnxzkCz-EdgSfPtx5ZaH1aroIdQdyn28nSsw3KkyzGP_CJO6mtPO3ySGNYexJXP7nz4-jPl-4DxhK_8Rncz4ZxBbMsLtI5mGnHb6X_dmC57hHlYz_Q1LjGys3-g0Y?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/N-UQvClxy3H5mmPKzydXGopkcIVAIJ3AJR1dyeJFPiyjTCyc0HjM58xzUEgF7sXc9czECCQDhJIX33calQ9dCQfzuDSKtsIDnPpCPc5veygBAeckd7TAuN8mBRtVxs8nMwnqC-KxnOj2Iskgy0hJQDYSiIdvCO2HAs5mjrt1_3q0ca09b4Rm4nUQUQTe8WpT?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/A4cc2ffdEVC8mz_-JlEoYq7A6HydZZto1LN0oFVsygcVntT1Xt9BSDkRP1NsUCpq2-wAhKeaIeTRFAQfS1MJ4h4OaCmNLmXfjmNZ64fQpVGlTXXvAcL6qKt1-PlGuVoG-bsqePfnYKhYxcJ_U6IgWeyRkxq-90qjLLhob_Hp7hYygOsjig_jI4-gVFulHVtD?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/teZaSh5bHBPhviOyxgnNq2Yl7q_sGuIA-7sf9W7CShWdg8L99q5aYe9UDXEAKKDb_IJIis8OXLYCll0Wiw_LX2i_cvcXFOMjrk8OTHnZLJGbmk2yb76upggHHmLFTxNgEssjRwEJaeus-UVEfK16enu9-uunyEmBKKR5Bl2zWbBGhC8Yxcshe7D-MVR7kakA?purpose=fullsize)

Task:

* Identify what is in an image

Examples:

* Cat vs Dog classification
* Recognizing objects like cars, trees, people

How ResNet50 helps:

* Extracts deep features from images
* Produces accurate predictions

Used in:

* Google Photos
* Instagram tagging systems

---

### 2. Object Detection

![Image](https://images.openai.com/static-rsc-4/NNUVAYZ9YEs6YPFR9o-10DHNQuf7IKiPCQh-Rq5eOOvW3ewh0_csxxu0IJLlLuIXQH6tokGbxvp-m55Pzlqfl4487UloZiSVKUiocGXbr5bgMX4jznwYHiuFOQX-nivs6WVotRBYQ4jnj_-TIgopF4FPUqx0aSeuNs2DrDto7X2x96hsvmBq3jn00cUpZsuj?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/vnnHSDoRQdn1ZZwjQ18JAXnZbqfACetfl0HWaI94LL3C7WZCbo3uhu8mCHiwD1RgRCAkPw5JrPO1Yo3d9MBYT-rJGD3JqGJdOW0_qXFki5loXUiYL-ARUz4FnmCoGZHRsHbzsNes5Cwmd7HrL7ZQl8WM1DHgbRimy4AvkCIrIK9m1IwpGxePLRdUjf03DERz?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/DubB8OWLa4EzaIelR2Br4tbL6e3ekfJ-SUOpBMEMfbpfqiZfDX9et53ddfe26BtwSHpW3s7-39v2zjXq20jIyRjzv2YQnRjIja0dByxzRnnAA1YUqjYRhF61VWW38Xxy6NMPA6z-ugpU8Duz-1SQwlpdhf_0yfOW_NAjE_Dojt2ryGqrjqQ9LUSdq52aV_Ua?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/rdNlcX80MP6Yk4r5AcT_dfIdEetgXOFFBBO4RQ3Zw2aJpvZkPqhTfuFCFLqcBJjpS77I7bjrqy6_OG0xnUJfpOX9eFvkPW9ijpKkTDHtcrpkYCQxYW_J_poXs6RawuGr6jTK_zpvfGYIY_GCYadzeeDOuTkzW-zsnm7x7qzF_7yD6zB8nxFum5ATqZLp1JzI?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/vHzetPrHezEJHWNxGT56Dnl6tHJ7ZN10cn2MHVMpH_YN5GW1G_gaDZI4pk0F4SpSY1JM_RutufjA2n7kAufpBQBtoOsJfXeeh6cvGTFXRUOX6QRiyMLBfOeZpBjyyB88nN4CATDZ6cgOfdan3x7cJVwmxQ9JE6WAVeDEcdGPTDrc5UKLxpsKBhKXmlnj24gf?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/ZAVOIH_MwEnwUjPTSdAQW0m5NdvqnuQj5sUz0eCHUSNxNmw9lhrLNZThZY0gADoL2gnx32BFEYmK_54xo3qHGnVy2xwOvQNqybFbto1kTYjarDINzxXG2dDfcZS64wxDgdkWARquDZHKmkcWwZqeccdkAIzU5UTkHpczqXbsIeh55AEct_T5GFZEbUXJEtXK?purpose=fullsize)

Task:

* Detect multiple objects and their locations

Output:

* Bounding boxes + labels

Examples:

* Detecting pedestrians in self-driving cars
* Security surveillance systems

How ResNet50 is used:

* Often used as a **backbone network** in models like Faster R-CNN

---

### 3. Medical Imaging

![Image](https://images.openai.com/static-rsc-4/SYwiB5Nr989s3T0W_JEwmcxYxcM0cACrqGaO4Heq2_LIINYpq06SJYnSpO9RJn0h2V_iscfRlHPEwsd89r9BNplMf8E5GaU6gs53P69IamXn_v4qQjcL_Hq5Dm_OnfMbB92WQx4micEa_kbHgKxaJnIUymaAcql3LoedjoexJWSvmx1ZpmHLU1yc-8AUGKoM?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/rRAt_3RuF5MDFOUcpDDgCRhJOht8Tccdxv6tjtFwwwaZc5HX4u17KwnYgdQzyR1yxU-90dDS953dvODdGkeQeu-hmEFe12v1Bya_7KrmTrgbrIQnjJyMiGzvMad4mtfJn9pOE96xmGFwg9aMdnoGS5kQIVLNGiCcMwBgs-2FtLZ7IoG5QYMcS0VmzSlg1wW4?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/7d0Miq3tpdatIUXjgsdh8BwF2iZo5cC4X874rsus7JT3Smz77KIedWbZvaPJSplsrZSYkjTOvc3iRR8IaDOfXorkHlENIF_HnkwtYKiWx_JOTNHLryUNC3gRJ4QzPclc_9v5zZvNDxEH-HYDSdYDVtRq4LRoTmNriyaVltedYUugh1tO4-t4MsrWHh0ntsLf?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/b9BXddaI2jLVq78CufNQvlc54hc27D_fuF6Ou12tlf3E7ss7o2hhJV8iPGHZ0JDqVZ3ycuWhcBiHLp52w2HLQWvvacRGP37IrAvqn_COH7lYTGzA05kL3YwN0E_PxWVP80n6sMcdJgFfoncoK16cJe6ODx1JIeAdfj8Am2Pr6o5MyCfMTEauZxKkNtnCr2vC?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/uG98ab8ynj2hY5mx17otqIikr_cRwZrimfOrp4TCkoPNmcr9L7Px7iA-oW9vzUAUrnEmHaW-_14BKFToCMgWPCCWgxTp2yucptu85vK5dUQq9_QH_2nARbgVHyd4GWJDRnCv6dgg4Rk6jxPyud1IAn40g8uwD8LiCTGMxHX9qMfiNSqFEPE-C5r3Xv7OJ84j?purpose=fullsize)

Task:

* Analyze medical images for diagnosis

Examples:

* Detecting tumors in MRI scans
* Identifying pneumonia in chest X-rays

Why ResNet50 is useful:

* Can capture very fine patterns
* Helps doctors in early detection

---





