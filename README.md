# Pneumonia_Prediction

```markdown
# Pneumonia_Prediction

This repository contains a deep learning project for classifying chest X-ray images into **Normal** and **Pneumonia** categories using convolutional neural networks. It includes datasets, training notebooks, and model artifacts.

---

##  Directory Structure

```

PneumoniaPrediction/
├── ChestXray/ # Root folder for dataset
│ ├── train/
│ │ ├── NORMAL/ # X-ray images labeled as normal
│ │ └── PNEUMONIA/ # X-ray images labeled as pneumonia
│ ├── val/
│ │ ├── NORMAL/
│ │ └── PNEUMONIA/
│ └── test/
│ ├── NORMAL/
│ └── PNEUMONIA/
├── pneumonia_prediction.ipynb # Jupyter notebook for model training and evaluation
├── trained_model.h5 # Saved Keras model after training
├── accuracy_vs_epochs.png # Training vs validation accuracy plot
├── loss_vs_epochs.png # Training vs validation loss plot
└── README.md # This documentation file

````

---

##  Project Overview

- **Goal**: Develop a CNN-based classifier to detect pneumonia from chest X-ray images.
- **Dataset**: Structured into `train`, `val`, `test` splits; each containing subdirectories for `Normal` and `Pneumonia`.
- **Notebook**: `pneumonia_prediction.ipynb` includes steps for data loading, preprocessing, model training, evaluation, and visualization.
- **Outputs**: A trained model (`trained_model.h5`) and visual performance metrics (`accuracy_vs_epochs.png`, `loss_vs_epochs.png`).

---

##  Setup & Usage

### Prerequisites

- Python 3.7+
- Libraries: TensorFlow/Keras, NumPy, Matplotlib, scikit-learn, Pillow / OpenCV
- (Optional) GPU-enabled environment for accelerated training

### Running the Notebook

1. Launch `pneumonia_prediction.ipynb` in Jupyter or Colab.
2. Execute all cells sequentially to:
   - Load and preprocess images
   - Build and train the CNN model
   - Validate performance and generate accuracy/loss plots
   - Evaluate on the test dataset

### Using the Trained Model

To load and infer with the trained model:

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('trained_model.h5')

# Example prediction:
img_path = 'ChestXray/test/NORMAL/your_image.jpeg'  # replace accordingly
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

prediction = model.predict(x)
if prediction[0][0] > 0.5:
    print("Pneumonia")
else:
    print("Normal")
````

---

## Results

* **Training Accuracy**: \~89%
* **Validation Accuracy**: \~88%
* **Test Accuracy**: \~89%

### Confusion Matrix (on Test Data)

|                      | Predicted Normal | Predicted Pneumonia |
| -------------------- | ---------------- | ------------------- |
| **Actual Normal**    | 158              | 12                  |
| **Actual Pneumonia** | 20               | 195                 |

*(Values shown are sample results; exact numbers may vary depending on random initialization and training runs.)*

* **Learning Curves**: Refer to `accuracy_vs_epochs.png` and `loss_vs_epochs.png` for detailed training performance.

---

## Future Enhancements

* Transfer learning (e.g., using pretrained networks like ResNet, EfficientNet)
* Hyperparameter tuning (learning rate, batch size, optimizer variations)
* Deploy model via Flask/FastAPI or convert to TensorFlow Lite for mobile use

---

## Dataset Access

If you would like access to the dataset, please contact me at **[charani.chukka@gmail.com](mailto:charani.chukka@gmail.com)**.

---


