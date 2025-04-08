# 🐾 AnimalRecognitionNN – Deep Learning for Animal Face Classification

This project was developed as part of the **Neural Networks Basics** course at the *Faculty of Automation Systems Engineering*. The objective was to design and implement a **convolutional neural network (CNN)** capable of classifying animal face images into one of three categories: **cats**, **dogs**, or **wild animals**.

The project includes a full machine learning pipeline, from dataset preprocessing and data augmentation to model training, custom metric tracking, and detailed visual analysis.

---

## 🎯 Objective

The task was to solve a **multi-class classification** problem using deep learning. Using the popular *Animal Faces* dataset from Kaggle, we aimed to:

- Detect and classify images of animal faces
- Implement a custom training strategy with callbacks and learning control
- Analyze model performance using detailed metrics and visualization tools
- Explore generalization to augmented/unseen images

> 🧠 What makes a cat different from a lion? With enough data and the right network, a model can learn the difference.

---

## 📦 Dataset Overview

The dataset contains 16,130 images split into:

- **Training set**: 13,130 images
- **Validation set**: 1,500 images
- **Test set**: 1,500 images
- **Augmented test set**: test images with transformations

All images are organized into three classes:

- 🐱 `cat/`
- 🐶 `dog/`
- 🦁 `wild/` – includes wild animals like lions, wolves, tigers

Each image was resized to **128x128 pixels**, converted to RGB, and normalized.

Data augmentation techniques (rotation, brightness shift, flip) were applied during training to improve generalization.

---

## 🧠 Model Architecture

The model was built using **Keras** and consists of:

- 🔸 **3 convolutional blocks**:
  - Conv2D → ReLU → MaxPooling2D
  - Increasing number of filters (32 → 64 → 128)
- 🔸 **Fully connected layers**:
  - Flatten → Dense(128) + Dropout
  - Output layer with 3 neurons (softmax)
- 🔸 **Loss Function**: Categorical Crossentropy
- 🔸 **Optimizer**: SGD with momentum and Nesterov
- 🔸 **Metrics**: Accuracy, Precision, Recall, F1 Score, MSE, ROC-AUC

## ⚙️ Custom Training Features

To improve performance and training control, we implemented:

### ✅ EarlyStopping
Stops training when validation loss doesn't improve for 3 consecutive epochs.

### 📉 LearningRateScheduler
Automatically reduces the learning rate by half when no improvement is detected.

### 🔁 Dynamic Batch Size
Batch size increases with each phase of training:  
`32 → 64 → 128 → 256`  
This accelerates convergence while avoiding early overfitting.

### 📊 Custom Callback
Custom metric tracking for:

- F1 Score  
- Precision / Recall  
- ROC-AUC  
- Mean Squared Error (MSE)  
- Weight distributions across layers  

All metrics are logged and visualized over time for analysis.

---

## 📈 Results & Evaluation

The model performed exceptionally well across all datasets. Summary of evaluation metrics:

| Metric     | Training | Validation | Test | Test (Augmented) |
|------------|----------|------------|------|------------------|
| Accuracy   | 96.96%   | **98.75%** | 98.13% | 95.87%           |
| Loss       | 0.0830   | **0.0373** | 0.0473 | 0.1057           |
| MSE        | 0.0149   | **0.0065** | 0.0086 | 0.0187           |
| Precision  | 97.13%   | **98.75%** | 98.20% | 96.25%           |
| Recall     | 96.87%   | **98.52%** | 97.93% | 95.80%           |
| F1 Score   | 97.00%   | **98.63%** | 98.06% | 96.02%           |
| ROC-AUC    | 99.78%   | **99.96%** | 99.90% | 99.65%           |

➡️ The model generalizes well even to **unseen and augmented data**, indicating strong robustness.

---

## 📊 Visual Analysis

The `plots/` folder contains a wide range of training visualizations:

- 📈 Accuracy and Loss per epoch  
- 📏 Precision, Recall, F1 Score trends  
- 💡 Learning rate and batch size tracking  
- 🔍 ROC-AUC curve  
- 🔬 Weight distribution across layers  
- ❌ Confusion matrix and misclassified samples

---

## 🔍 Inference – Single Image

Run prediction for a custom image:

```bash
python test_single.py --image path/to/image.jpg
```
The model outputs class probabilities and the predicted label for the image.

---

## 🧪 Key Learnings

- Data augmentation and dynamic batch sizing enhanced generalization  
- Custom callbacks gave deeper insight into model behavior  
- Learning rate and weight monitoring stabilized training  
- The CNN was efficient, lightweight, and achieved strong accuracy with relatively few epochs

---

## 👥 Project Information

This project was developed as a **group assignment** for the course **Neural Networks Basics**.

- 🎓 Degree: *Automation Systems Engineering*  
- 🗓️ Academic Year: 2024/2025  
- 👨‍👩‍👧 Team: 3 students (Wiktoria Michalska, Grzegorz Cyba, Patryk Kurt)

---

## 🐶 Dataset

The dataset used in this project can be downloaded from:  
🔗 [https://www.kaggle.com/datasets/andrewmvd/animal-faces](https://www.kaggle.com/datasets/andrewmvd/animal-faces)

After downloading, the data should be manually organized into the following folder structure:

- `dataset/train/`
- `dataset/val/`
- `dataset/test/`

Each of the above folders must contain **three subfolders**:

- `cat/`
- `dog/`
- `wild/`

📌 This structure is required for proper use with Keras’ `ImageDataGenerator`.

  

