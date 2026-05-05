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

## Understanding Residual Learning

Residual learning is the core idea behind **ResNet**. Instead of forcing a deep network to learn a complete transformation from input to output, it learns only the **difference (residual)** between them.

### Traditional Learning vs Residual Learning

* **Traditional approach:**
  The network tries to learn a direct mapping
  ( H(x) )

* **Residual approach:**
  The network learns a smaller function
  ( F(x) = H(x) - x )

Then combines it with the original input:
[
H(x) = F(x) + x
]

### Intuition

It is easier to learn **small corrections** than to learn the entire transformation from scratch.

Example idea:
If the correct output is very similar to the input, the network only needs to adjust slightly instead of rebuilding everything.

---

## Concept of Skip Connections

A **skip connection** (also called a shortcut connection) allows the input to bypass one or more layers and be added directly to the output.

### How it works

1. Input ( x ) goes through some layers → produces ( F(x) )
2. The original input ( x ) is added directly to ( F(x) )
3. Final output becomes ( F(x) + x )

This “shortcut” creates an alternative path for information and gradients.

### Why it is important

* Prevents information loss
* Allows gradients to flow directly backward
* Makes training deep networks stable

---

## Identity Mapping

**Identity mapping** means passing the input forward **unchanged**.

In ResNet:

* The skip connection carries the input ( x ) directly
* This acts as an identity function

So even if the main layers fail to learn anything useful:
[
Output = x
]

### Why this is powerful

* Guarantees that deeper layers will not perform worse than shallower ones
* Makes it easier for the network to learn or “do nothing” when needed

---

## Mathematical Intuition

The core equation of residual learning is:

H(x) = F(x) + x

### What each term means

* ( x ): input
* ( F(x) ): residual (what the network learns)
* ( H(x) ): final output

### Why this helps

Instead of learning:

* A complex function ( H(x) )

The network learns:

* A simpler function ( F(x) )

If the optimal mapping is close to identity:

* Then ( F(x) \approx 0 )
* So ( H(x) \approx x )

This is much easier for optimization.

---

## Why Skip Connections Help Training Deeper Networks

Deep networks face two major issues:

* Vanishing gradients
* Degradation problem

Skip connections directly address both.

### 1. Better Gradient Flow

During backpropagation:

* Gradients can flow through the shortcut path
* They do not shrink as much as in long layer chains

Result:

* Early layers learn effectively
* Training becomes faster and more stable

---

### 2. Avoids Degradation Problem

Without skip connections:

* Adding more layers can reduce accuracy

With skip connections:

* The network can simply pass input forward
* It will not perform worse than a shallow network

---

### 3. Easier Optimization

Learning ( F(x) ) is simpler than learning ( H(x) )

* Small residuals are easier to adjust
* Optimization landscape becomes smoother

---

### 4. Feature Reuse

* Earlier features are directly reused
* Later layers refine instead of rebuild

---

## Visual Explanation of Residual Block

![Image](https://images.openai.com/static-rsc-4/jj5HH4gfaz_Oy8Q8j4nabTYaFTOra3sK077wULSkF5pW2dHW5TcxE6vI67AYfHZyTxYnLHL_onwn0IuTAkYhEeoypdeQgUWGVrhPDZDMvUGNSJxHMp1smY-4OXIUBPv57mxjs7IqcnvN4ww2X8V2WIc-ZMCU2I0MLfwrfXmrkZoSKUa7QJDFUSpZb-6iI5xx?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/IgfhLTNt1ZUk1uMj7SzPgSqSqm5lKcakmMWapVxcTVv6pgp-3D7ge6bRWuqHxpAefpsmtAwo3eL8r8YsQ2Pjbrw74nrK-GfsobY4EdapG3u9NgSWlpGswq878W862jR-5Oh4qG-oUvGymy9Pn4h7OCa9B9jbtZZR360WzSh_EPU1vjufs91qZPeiQHQtVUGP?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/3YS3DjGyf7a55O3kjsMnyM42vk0QcXD_fwMaRRQZD8Sa-gLw8ACJ-_19xix0tuinh4jcsXhktBaZ7h1J_Db8mZmAzu-k9SxxX2R71DYHuGza-tSJhGOnQt6YPocKSjWcrMKcjqCrUfjDxMQtA-if3WsWRw9GNXsfkgeg_lgcTUWClF_OHr6TM62qJY0bahoU?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/lOfVqO9_ApC255VwfyPFJJ4NEUIyFKZY5M3WY2BtLcK56dURjByCfLA8GsyAetILFnZUrAHPsR7WKAB3yg5HiGHOP8lmIX_uU1XCmfSatU4ECeI0PZwCvvlupwu9oieGfsTiDZUoGM1fGCv06K0wtiF8Vj5GZNFRRpm_E9gALMDTC9Z_sIAYbWzekguxxWNu?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/PSspmrGdCIINVgT_4SIumJLV9BuN9953Ny6eyghMYUdEGat09mT7-f6b-NKamOrnE64onAHm66GkQF90UYk5sTfeX1Z3mHyx_5vQ_rjiZqA30LRmBwdgmWTLA0drJPISIfrxHvy1rycqV6kz3SHQV-exzHkCzQCOTNy263nS0C_XeRqRQRrmpHuL48NZteNM?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/C04-9ZrBsfF6MNUQPQe5i4mKLBdHMgOFg5NenlJe04auIeoQXJLcLd2zhjg__UR9gPJfTjqO7yITbV6Du_kjRUFSMrJxTmDisSlX-srYH6mwV0-v2Q75lZzfvP1JXEHiOQWAeyrfcrdi5WHnPUdlh2AO7ba4Yc4s6VznbAgXP50haJhTaIrXRLvQj0ievHWr?purpose=fullsize)

### Step-by-step flow inside a residual block

1. **Input (x)**
   The original data enters the block

2. **Main Path (F(x))**

   * Convolution layer
   * Batch Normalization
   * Activation (ReLU)
   * Another convolution

3. **Shortcut Path**

   * Direct connection carrying input ( x )

4. **Addition**

   * Combine both paths:
     [
     F(x) + x
     ]

5. **Final Activation**

   * Apply ReLU to the result

---

## Architecture of ResNet50

**ResNet50** is built by stacking **residual blocks** in a very organized way. Understanding its architecture becomes easy if you break it into two parts:

1. Overall flow of the network
2. Detailed layer-by-layer breakdown

---

# 4.1 Overall Structure

![Image](https://images.openai.com/static-rsc-4/i-B0VY6mDkAJTGi5TQBnvKu7Xw2vmBl-nOtHN6xtxhUUxDUlicEWzwJQpz3f3_ha7ByLrpvQzeK-m2VJX6hdd8mEF5bvSJQN9RW-cGVdvAnAc2_EkX6pLxhwZfw0vAuGlKbZXbvIENSwOABIRk65QwVzkFr_FB_7L37WlHj4Cp1pEwvgn3QzmFyaGYhG2cxL?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/OvSG0xYRgwBAIvzLypTudCEdSS-1JITiFYeVIxFy9SWQ_S2U1_6547fD-Vc9DKWmW4p-devjkcSNgytVJP5BGdX8Jg-RV9CDGahsxFxOMRkf-HJvrlB9uIRbTeITtBO2i4Eu8jv1qlQow0fQOqry1KbMKdhALCV-qBUeqhPBFaGl4nGuYOMNxcco0kN_nKXc?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/t1KgXmpY_FqaUbF8ANhhLqYqqMjv0gIJAX4MZDb9OcePgDWrP_O8Nc6iZ2TiDyCh4GD0dk5brASz6_1JWMNaw11GJy0uR4JvedNVS6HegC92kqdHXbrnxe8HIohPfpUvdwNz7EW0bGbuubLd5NrgnaMoaTdiwUTGzK3147K-MwST5EjBLkhD08mjwjQe7xEd?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/2nM1p-B7RxHVwgS2QXjvMBfloe9kJ-vnsG50qLFBCi1eJNTMGU2OmB3DzGrrQxdxEQlCfH5BKr6uKGYdwdty_Cv-51v4j98LwCmkdN0Q7POOLuqC18xA8oeeNtwgexSHkrEYWboc8rxN6vUrgVAX1_qSE8YCKvGdf2Hrzo1wt_lwtONQlHuXNEGCUzdbs5fD?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/0-aZnbQHT0HdK1_PeppdSIma7iRxmw5WgI5oY-MeCph1yOF_1EKmxCHGhRpVnqSTBonPLc225batoyBX_Zel6MN2JuqYs3ivgftctf3ndlxM2wEPe7iLYMkw4qt1q_epg9MTUNmHlJqtdKGHD7u0Q8lDtTBO8bea9ICVMiou1ECJRoP6RjL3jPbozbvMolqW?purpose=fullsize)

### 1. Input Layer

* The model takes an image as input
* Standard input size: **224 × 224 × 3 (RGB image)**

This is just raw pixel data.

---

### 2. Convolution + MaxPooling

This is the **initial feature extraction stage**

* **7×7 Convolution**

  * Large filter to capture basic patterns
  * Output feature maps

* **Batch Normalization + ReLU**

  * Normalize values
  * Introduce non-linearity

* **MaxPooling (3×3)**

  * Reduces spatial size
  * Keeps important features

Result:

* Image size becomes smaller
* Important patterns like edges and textures are extracted

---

### 3. 4 Stages of Residual Blocks

This is the **core of ResNet50**

There are 4 main stages:

* Conv2_x
* Conv3_x
* Conv4_x
* Conv5_x

Each stage:

* Contains multiple residual blocks
* Learns increasingly complex features

Flow of learning:

* Early stage → edges, colors
* Middle stage → shapes, textures
* Deep stage → objects

---

### 4. Fully Connected Layer

* After feature extraction, output is flattened
* Passed to a dense (fully connected) layer

Purpose:

* Combine all learned features into final decision

---

### 5. Output Layer (Softmax)

* Final layer uses **Softmax activation**

Output:

* Probability for each class

Example:

* Cat → 0.92
* Dog → 0.05
* Car → 0.03

---

# 4.2 Layer Breakdown (Detailed Structure)

Now let’s break down how those 50 layers are arranged.

---

### Conv1 Layer

* 7×7 Convolution, 64 filters, stride 2
* Followed by:

  * BatchNorm
  * ReLU
  * MaxPooling

Purpose:

* Capture basic visual features

---

### Conv2_x (3 Blocks)

* Contains **3 residual blocks**
* Output feature size: relatively large spatial resolution

What it learns:

* Simple patterns like edges and corners

---

### Conv3_x (4 Blocks)

* Contains **4 residual blocks**
* Spatial size reduces, depth increases

What it learns:

* Shapes and textures

---

### Conv4_x (6 Blocks)

* Contains **6 residual blocks**
* This is the **deepest and most important stage**

What it learns:

* Complex object parts

---

### Conv5_x (3 Blocks)

* Contains **3 residual blocks**
* Very deep representation

What it learns:

* High-level object understanding

---

### Total Layers Count

* Conv layers + FC layer = **50 layers**
* That is why it is called **ResNet50**

---

# 5. Types of Blocks in ResNet50

ResNet50 mainly uses two types of residual blocks:

---

## 5.1 Identity Block

![Image](https://images.openai.com/static-rsc-4/LMSAiPf65zu0x5p3ugHDzLBdxBIEv9G3IEFT5C0X0KwOrsvN0mSwuhEUBtw0X7YnP8O4DArCaa4Oh18PfeBvBR5NtTd5YrJd0hJCs7op5tHW9z1ImFnxRhszKD4YP1SWEDlO6vF3xSZjdR3xsYtsZxJsSgQ06HojtG83Qe3tYwtBst5KkPD7lwdDfhHEj0Rt?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/lOfVqO9_ApC255VwfyPFJJ4NEUIyFKZY5M3WY2BtLcK56dURjByCfLA8GsyAetILFnZUrAHPsR7WKAB3yg5HiGHOP8lmIX_uU1XCmfSatU4ECeI0PZwCvvlupwu9oieGfsTiDZUoGM1fGCv06K0wtiF8Vj5GZNFRRpm_E9gALMDTC9Z_sIAYbWzekguxxWNu?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/H22vdb-7tiEVJ_IduM_bk_KEXlwbzpzJ0m3Ma-Le8mQbXmsUkq89NoCA43nDzhnqSIGF1wr0Q9ID0nPs6eiLIkTi2ixMNCNDtP1S0VpIZUrfOj-W-x5Y_TJqa6OJKt_hbMIVSZP5G2a6_zQ90VI1Hoj7KuWr9FBy59bkeRGeCMZs_rC6sRVVL7PXV6Q2iDkj?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/_h0GGb7M1AfY_rZ6U_6Lfd7Tcmok_gPiTyh17P9ki1MquEYgcSRcKpmTY0VwK2L0-8RM5iyd2hpTZg-H_iZdzONi8oAUFd__Q51KMlLe68z_F0bFWG_OYiMLlraLtU3HOE7DdIyT3pXLVpXVk_TK0c8G40yR2SDZtutlSW0O4RfZbIGm6XZ_RiM47w5IwgVm?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/QNNz2SESTzeb96fW5bMSpJ1sMN3nncW3ewFTwdc2wzDcIH3B6mQ8m-vWy9VDlAOIVtq4pskWrvXYR0H830h5bWRm1Snv1d4FY1UXAQP__i2LyufdTY3rmOjmHbBVBLk7dnDxVDkQv5RExdtl89JRUcasyKh50tcW9eAlKFIFcMzkUNrWlK_FM96WPzVHhCSN?purpose=fullsize)

### When it is used

* Input and output dimensions are **the same**

### Structure

* Input → Convolution layers → Output
* Shortcut path → direct (no change)

### Key point

* Shortcut is just **identity mapping (x)**
* No convolution is applied in shortcut

### Output

[
F(x) + x
]

### Why useful

* Simple and efficient
* Keeps information intact

---

## 5.2 Convolutional Block

![Image](https://images.openai.com/static-rsc-4/1hNUTee1RgefipUeq4wDZlfpX9fypXnSDjF1H_otpJM2YiQtyx3hltbF2l8n1oDN7wDnmwybZxXA3DCo3NSIno-4MKbv0-UH93ZVQFdEF-yhsaClcxdFbkPS7T5eB2EgUqm59zLMtuWzbSO--yfDuckzZD_axoFUuBbKkXZZFffVD7XeogVQYRiOoSftFvvo?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/iUIkIrXs0mLjaB_soJtCxj9_ltXeUOm7jliUz4oaO1lirdoPInALwQovFqoYT_fcAcOtcXuOHQVgx-vcqtSaG59iG4A6NszYAfuu6xhE49anqHqqiFTJwP6BgvzGpXoB-LztfxzUhAadGRfcl1MJkbRDAmG-Zz01obWDU-xQgy1ITODw_6RkziEPGhTzg2dK?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/5zBPSWHOcdg00epOwAiJi49bb53r_qp2H974JCPaUUabtqAlWiUrppQVsy-koD0BozZxV423IHLqxDYcSqcEfomVSM-XclTVZxnaJiCLCcZt9uTrl_pbObMyClxL7qA09eG186rNMUdi489r0FicMdPTc0fz40-XmCOsxQjb3pwTRln6BtOpjU8Zjn-kgMHh?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/r-HyXVYIcEhoUfWQ_FFr0BTuJF3yb82C9ayYvnPVKtB3tUqRUGzy-8xiCLc-Fzu0j0npD9usvNUHKyNzn5YzSGfQvHumf-je6CQErEvxhp2Q4tNyaVjLKXLg2nM_OZmT6bGkPKngceval7ev4kmWk5lj8GqWDxZNQyxBnod9mCSuyM5MPv52XYgm4MsNi40C?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/0ZzBk7QRXnkRhAJYHMP05onN5U5Ljtp2XI_jSKqiUMYMFzGTmqx__uXXSAUaODauSOo594U412epaeFTiF0fOW86VJPjoK6xB-_j40kkuV-tbwKpT3fE7apOUXtNIVsFdcB5rt_xGD1I1oDKda0WXclP14ZCnZPO3RwsQ_fIrPCKoO2ng7lTvRcjlVzsjAvG?purpose=fullsize)

### When it is used

* Input and output dimensions are **different**

Example:

* Feature map size changes
* Number of channels increases

### Structure

* Main path: Convolution layers
* Shortcut path: **1×1 convolution**

### Why convolution in shortcut?

* To match dimensions before addition
* Ensures shapes are compatible

### Output

[
F(x) + W_s x
]

(where ( W_s ) is convolution in shortcut)

---

## Key Difference Between Blocks

| Feature    | Identity Block | Convolutional Block       |
| ---------- | -------------- | ------------------------- |
| Dimensions | Same           | Different                 |
| Shortcut   | Direct         | Convolution               |
| Complexity | Simple         | Slightly complex          |
| Usage      | Most blocks    | First block of each stage |

---

## Simple Intuition

* **Identity Block**: “Keep same size, just refine features”
* **Convolutional Block**: “Change size, learn new representation”

---

## Final Summary

* ResNet50 is built using **stacked residual blocks**
* Architecture flows as:

  * Input → Conv → Pool → Residual Stages → FC → Softmax
* 4 main stages:

  * Conv2_x → Conv5_x
* Two block types:

  * Identity Block (same size)
  * Convolutional Block (dimension change)

---







