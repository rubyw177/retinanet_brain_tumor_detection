# RetinaNet Brain Tumor Detection

## Overview
This project implements a **brain tumor detection model** using **RetinaNet** with **transfer learning** in **TensorFlow**. The model is designed to detect tumors in **MRI brain scans** with high accuracy by leveraging a pre-trained object detection network.

## Features
- **RetinaNet-based detection**: Uses transfer learning for robust tumor identification.
- **Custom Training Loop**: Implements TensorFlow's flexible training pipeline.
- **Bounding Box Predictions**: Localizes tumors within MRI images.
- **Efficient Inference**: Optimized model for real-time detection.

## Dependencies
Ensure you have the following installed:
```bash
pip install tensorflow numpy pandas matplotlib opencv-python albumentations
```

## Usage
### 1. Load Dataset
The dataset should contain **MRI images** labeled with **bounding boxes** for tumors. The images are preprocessed before training.

```python
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load dataset
image_path = "dataset/image1.png"
image = load_img(image_path, target_size=(512, 512))
image_array = img_to_array(image) / 255.0
```

### 2. Define RetinaNet Model
The RetinaNet model is built using a **ResNet50** backbone for feature extraction.
```python
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# Load pre-trained ResNet50
base_model = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(512, 512, 3)))
```

### 3. Train the RetinaNet Model
Run the following command to train the model:
```python
python Brain_Tumor_Detection_RetinaNet.ipynb
```

The script will:
1. **Load and preprocess MRI scans**.
2. **Apply RetinaNet's feature pyramid network (FPN)**.
3. **Train with custom data augmentation techniques**.
4. **Evaluate performance on test images**.

```python
from tensorflow.keras.optimizers import Adam

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=50, batch_size=16)
```

### 4. Perform Inference
After training, you can use the model to detect tumors on new MRI scans:
```python
predictions = model.predict(np.expand_dims(image_array, axis=0))
```

### 5. Visualize Predictions
Bounding boxes and confidence scores are plotted over test images:
```python
import matplotlib.pyplot as plt
plt.imshow(predicted_image)
plt.show()
```

## Implementation Details
- **Backbone Model**: RetinaNet with ResNet50 as the feature extractor.
- **Loss Function**: Focal loss to handle class imbalance in tumor detection.
- **Data Augmentation**: Applied **random flipping, brightness adjustment, and Gaussian noise**.
- **Evaluation Metrics**: Intersection-over-Union (IoU) and Precision-Recall curve.
