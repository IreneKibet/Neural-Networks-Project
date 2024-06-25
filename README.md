# Neural-Networks-Project
# Diagonal Integration for Multi-modal Prediction

This project focuses on integrating information from two distinct data modalities using a neural network-based approach. The objective is to accurately predict the test data for Modality 2, emphasizing the separation of features and samples between modalities and batches.

## Overview

### Data Preprocessing and Analysis
- **Data Loading**: 
  - Modality 1: `modality1_train.csv`, `modality1_test.csv`
  - Modality 2: `modality2_train.csv`
- **Data Dimensions**:
  - Modality 1: 2438 features, 1350 samples
  - Modality 2: 2438 features, 1350 samples
- **Conversion to PyTorch Tensors**: Data loaded and converted for use in PyTorch models.

### Model Implementation
- **Architecture**:
  - Two-channel neural network using autoencoder-based integration.
  - Separate encoders and decoders for each modality.
  - Shared autoencoder to combine encoded representations.

### Model Training and Evaluation
- **Training**:
  - Utilized Adam optimizer and Mean Squared Error (MSE) loss function.
  - Incorporated L1 regularization, learning rate scheduler, and early stopping.
- **Performance**:
  - Initial MSE reduced significantly over epochs, indicating effective learning.
  - Evaluated using modality1_test.csv as ground truth.
#### Technologies Used
- **Python**: For data preprocessing and model implementation.
- **PyTorch**: For building and training the neural network.
- **Pandas**: For data manipulation and loading.
- **NumPy**: For numerical operations.
