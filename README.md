# Implementing Custom Loss Functions in TensorFlow

As a deep learning practitioner, I created this project to explore and implement custom loss functions in TensorFlow/Keras, focusing on handling class imbalance and preventing overconfidence in predictions. The notebooks demonstrate my ability to extend Keras' Loss class for specialized scenarios, such as multi-class focal loss with regularization and a modified MSE loss. These implementations are intentionally designed to illustrate core concepts and mechanics, prioritizing learning over peak performance metrics, which serves as a benchmark for understanding loss customization in practice.

## Problem Statement and Goal of Project

Standard loss functions like cross-entropy can struggle with imbalanced datasets, where models may overfit to easy samples or become overconfident. The goal here is to implement a Categorical Focal Loss for multi-class classification, incorporating L1 and L2 regularization to address imbalance (e.g., in object detection), and a custom MSE loss that penalizes predictions deviating from 0.5 to discourage overconfidence and reduce overfitting in logistic-style outputs.

## Solution Approach

- **Custom MSE Loss**: Defined a class `custom_mse` inheriting from `tf.keras.losses.Loss`. It computes mean squared error plus a regularization term based on the squared distance from 0.5, scaled by a factor (default 0.1).
- **Categorical Focal Loss**: Implemented `CategoricalFocalLoss` with parameters for alpha (class balancing), gamma (focusing on hard samples), and weights for L1/L2 regularization on predictions. The loss clips predictions to avoid log(0), computes the focal term, sums across classes, and adds regularization.
- **Model Integration**: Built a simple CNN with convolutional layers (16, 32, 64 filters), max pooling, flattening, and dense layers. Compiled with the custom loss and Adam optimizer.
- **Data Handling**: Loaded and preprocessed the dataset, resizing images to 224x224, normalizing, and one-hot encoding labels.

## Technologies & Libraries

- TensorFlow (versions 2.19.0 and 2.10.0) and Keras for loss implementation and model building.
- tensorflow_datasets for dataset loading.
- NumPy (version 2.1.3) for general computations.
- Additional checks for GPU availability using `tf.config.list_physical_devices`.

## Description about Dataset

The 'cats_vs_dogs' dataset from tensorflow_datasets is used, consisting of images for binary classification (cats and dogs). Images are resized to 224x224x3, normalized to [0,1], and labels are one-hot encoded into 2 classes.

## Installation & Execution Guide

1. Ensure Python 3.9+ or 3.10+ is installed.
2. Install dependencies:
   ```
   pip install tensorflow tensorflow-datasets numpy
   ```
3. Download the notebooks and run:
   ```
   jupyter notebook "Categorical Focal Loss.ipynb"
   jupyter notebook "mini losscustom.ipynb"
   ```
4. For GPU support, ensure CUDA is configured (code includes checks for GPU detection).

## Key Results / Performance

- The model is fitted for 10 epochs on the preprocessed dataset using the custom focal loss.
- No explicit performance metrics are outputted in the code, as the focus is on loss implementation and training setup. This serves to demonstrate the mechanics of custom losses in action, with intentional simplicity for educational purposes.

## Screenshots / Sample Outputs

Sample code for custom MSE loss definition:
```
class custom_mse (tf.keras.losses.Loss):
    def __init__ (self , facture = 0.1 , name = 'custom mse'):
        super(custom_mse,self).__init__(name = name)
        self.facture = facture
        
    def call (self , y_true , y_pred):
        mse = tf.math.reduce_mean(tf.math.square(y_true - y_pred))
        reg = tf.math.reduce_mean(tf.math.square(0.5 - y_pred))
        return mse + reg * (self.facture)
```

Focal Loss formula from markdown:
```
FL(y_{true}, y_{pred}) = - α * y_{true} * (1 - y_{pred})^γ * log(y_{pred})
l1(y_{true}, y_{pred}) = ∑ |y_{pred}|
l2(y_{true}, y_{pred}) = ∑ (y_{pred})^2
total.loss = FL + l1_w * l1 + l2_w * l2
```

GPU check output example:
```
TensorFlow Version: 2.10.0
Num GPUs Available: 1
TensorFlow is using GPU: True
```

(For training logs, run the notebook interactively.)

## Additional Learnings / Reflections

Through these notebooks, I delved into the internals of focal loss, understanding how gamma focuses on hard samples and alpha balances classes, as explained in the provided article. The custom MSE highlights regularization to prevent overconfidence, with Persian comments noting its role in avoiding overfitting. GPU detection code reinforced my skills in hardware-accelerated training. These experiments underscore my grasp of loss customization, even with basic models, as a foundation for more complex applications.

## 👤 Author

## Mehran Asgari
## **Email:** [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com).
## **GitHub:** [https://github.com/imehranasgari](https://github.com/imehranasgari).

---

## 📄 License

This project is licensed under the Apache 2.0 License – see the `LICENSE` file for details.

💡 *Some interactive outputs (e.g., plots, widgets) may not display correctly on GitHub. If so, please view this notebook via [nbviewer.org](https://nbviewer.org) for full rendering.*