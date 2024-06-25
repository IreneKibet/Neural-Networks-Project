#!/usr/bin/env python
# coding: utf-8

# # Diagonal Integration for Multi-modal Prediction

# ### Importing the necessary libraries

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader


# ## Data Preprocessing and Analysis
# ### Loading the data
# 
# The datasets consists of two distinct modalities, each represented by separate datasets. For Modality 1, training data is split into two batches,each containing 1350 samples and 2438 features. The test data for Modality 1, serves for evaluation and tuning purposes. Modality 2 is represented by modality2_train.csv for training and modality2_test.csv as the primary prediction target. The objective is to accurately predict the test data for Modality 2, emphasizing the separation of features and samples between modalities and batches.

# In[2]:


# Loading the csv files.
modality1_train = pd.read_csv('modality1_train.csv', index_col=0)
modality2_train = pd.read_csv('modality2_train.csv', index_col=0)
modality1_test = pd.read_csv('modality1_test.csv', index_col=0)


# In[3]:


#Checking the dimensions of the data
print(modality1_train.shape)
print(modality2_train.shape)
print(modality1_test.shape)


# In[4]:


# Convert data to PyTorch tensors
features_modality1_train_tensor = torch.FloatTensor(modality1_train.values.T)
features_modality2_train_tensor = torch.FloatTensor(modality2_train.values.T)
features_modality1_test_tensor = torch.FloatTensor(modality1_test.values.T)


# In[5]:


# Checking the dimensions of our pytorch tensors
print(features_modality1_train_tensor.shape)
print(features_modality2_train_tensor.shape)
print(features_modality1_test_tensor.shape)


# ## Model Implementation and Evaluation

# I started with a simple neural network architecture designed for integrating information from two channels, Channel 1 and Channel 2, using an autoencoder-based approach. The model consists of separate encoders for each channel, which capture the relevant features, and a shared autoencoder that combines the encoded representations. Each channel has its own decoder to reconstruct the original input. The shared autoencoder is intended to capture shared or integrated information between the two channels.

# In[6]:


class DiagonalIntegrationNet(nn.Module):
    def __init__(self, input_size):
        super(DiagonalIntegrationNet, self).__init__()

        # Channel-specific encoder for Channel 1
        self.channel1_encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Channel-specific encoder for Channel 2
        self.channel2_encoder = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Shared autoencoder
        self.shared_autoencoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )

        # Channel-specific decoder for Channel 1
        self.channel1_decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, input_size)
        )

        # Channel-specific decoder for Channel 2
        self.channel2_decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, input_size)
        )

    def forward(self, x1, x2):
        # Encode each channel
        encoded1 = self.channel1_encoder(x1)
        encoded2 = self.channel2_encoder(x2)

        # Combine and pass through shared autoencoder
        combined = torch.cat([encoded1, encoded2], dim=1)
        shared_encoded = self.shared_autoencoder(combined)

        # Decode back to each channel
        decoded1 = self.channel1_decoder(shared_encoded)
        decoded2 = self.channel2_decoder(shared_encoded)

        return decoded1, decoded2


# #### Defining the function to train the model

# In[7]:


import matplotlib.pyplot as plt

# Training the model
def train(model, train_loader, optimizer, loss_fn, epochs):
    mse_values = []  # List to store MSE values for each epoch

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_data in train_loader:
            # Unpack batch data
            features_modality1, features_modality2 = batch_data

            # Forward pass
            predictions_modality1, predictions_modality2 = model(features_modality1, features_modality2)
            
            # Compute the loss
            loss_modality1 = loss_fn(predictions_modality1, features_modality1)
            loss_modality2 = loss_fn(predictions_modality2, features_modality2)
            
            # Total loss
            total_loss += (loss_modality1 + loss_modality2).item()

            # Backward pass and optimization
            optimizer.zero_grad()
            (loss_modality1 + loss_modality2).backward()
            optimizer.step()

        # Calculate average loss for the epoch
        average_loss = total_loss / len(train_loader)
        mse_values.append(average_loss)

        # Print the MSE for the epoch
        print(f"Epoch {epoch + 1}/{epochs}, MSE: {average_loss}")

    # Plot the MSE values
    plt.plot(range(1, epochs + 1), mse_values, marker='o')
    plt.title('Training MSE Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.show()


# ### Model Training and Evaluation
# 
# In this step we will initialize the model with the basic architecture and evaluate how the model perform then implement some tuning steps afterwards.

# In[8]:


# Initialize model, optimizer, and loss function
input_size = 2438
model = DiagonalIntegrationNet(input_size)
optimizer = Adam(model.parameters())
loss_fn = nn.MSELoss()


# In[9]:


# Creating a list of the training tensors
train_data = list(zip(features_modality1_train_tensor, features_modality2_train_tensor))
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

#Training the model
train(model, train_loader, optimizer, loss_fn, epochs=20)


# #### Model performance

# In[10]:


# Setting the modality1_test.csv as the ground truth for model evaluation
ground_truth_modality1_test_tensor = features_modality1_test_tensor

# Evaluate the model on modality1_test.csv
with torch.no_grad():
    model.eval()
    predictions_modality1_test, predictions_modality2_test = model(features_modality1_train_tensor, features_modality2_train_tensor)

# Compare predictions with ground truth for modality1_test.csv
mse_loss_modality1_test = torch.nn.functional.mse_loss(predictions_modality1_test, ground_truth_modality1_test_tensor)
print(f'Mean Squared Error on modality1_test: {mse_loss_modality1_test.item()}')


# The model, as indicated by the Mean Squared Error (MSE) of 0.3188 on modality1_test, performs well in minimizing the difference between its predicted values and the actual values for modality 1 test during testing. The model appears to effectively reconstruct or predict the samples from modality 1 test. 

# In[11]:


#Checking the dimensions of the predictions to ensure it is correct
predictions_modality2_test.shape


# In[12]:


# Changing the predictions tensor into a dataframe and transpose
predictions_modality2_test_df = pd.DataFrame(predictions_modality2_test.T)

#predictions_modality2_test_df.to_csv('predictions_modality2_test.csv', index=False)


# In[13]:


#Assigning the column names and the row names to the predictions
predictions_modality2_test_df.columns = modality1_train.columns
predictions_modality2_test_df.index = modality1_train.index
#Checking the shape of the predictions
print(predictions_modality2_test_df.shape)


# ### Transforming the predictions data into Kaggle submission format

# In[14]:


# Reshape data from wide to tall format
df_melted = predictions_modality2_test_df.melt(var_name='sample', value_name='value')

df_melted['id'] = np.arange(1, len(df_melted) + 1)
df_melted['id'] = 'id_' + df_melted['id'].astype(str)

submission = df_melted[['id', 'value']]


# In[15]:


# Saving the predictions in a csv file
submission.to_csv('mod2_test_pred.csv', header=True, index=False)


# ## Model Tuning and Conclusion

# ### Hyperparameter Tuning
# In this step, I added L1 regularization to the loss function and a learning rate scheduler that reduces the learning rate when the validation loss stops decreasing. I also added an early stopping mechanism that stops training if the validation loss doesn’t decrease for a certain number of epochs.

# In[16]:


import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Training the model
def train(model, train_loader, optimizer, loss_fn, epochs, lmbda, patience):
    mse_values = []  # List to store MSE values for each epoch

    # Add a scheduler for learning rate reduction
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    # Early stopping parameters
    best_loss = None
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_data in train_loader:
            # Unpack batch data
            features_modality1, features_modality2 = batch_data

            # Forward pass
            predictions_modality1, predictions_modality2 = model(features_modality1, features_modality2)
            
            # Compute the loss
            loss_modality1 = loss_fn(predictions_modality1, features_modality1)
            loss_modality2 = loss_fn(predictions_modality2, features_modality2)
            
            # Regularization
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            total_loss += (loss_modality1 + loss_modality2 + lmbda * l1_norm).item()

            # Backward pass and optimization
            optimizer.zero_grad()
            (loss_modality1 + loss_modality2 + lmbda * l1_norm).backward()
            optimizer.step()

        # Calculate average loss for the epoch
        average_loss = total_loss / len(train_loader)
        mse_values.append(average_loss)

        # Learning rate reduction
        scheduler.step(average_loss)

        # Early stopping
        if best_loss is None or average_loss < best_loss:
            best_loss = average_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping")
                break

        # Print the MSE for the epoch
        print(f"Epoch {epoch + 1}/{epochs}, MSE: {average_loss}")

    # Plot the MSE values
    plt.plot(range(1, len(mse_values) + 1), mse_values, marker='o')
    plt.title('Training MSE Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.show()


# In[17]:


# Add a scheduler for learning rate reduction
scheduler = ReduceLROnPlateau(optimizer, 'min')

# Regularization strength
lmbda = 0.01

# Early stopping parameters
patience = 10
best_loss = None
early_stop_counter = 0


# In[18]:


# Train the model with the new parameters
train(model, train_loader, optimizer, loss_fn, epochs= 20, lmbda= lmbda, patience = patience)


# ### Conclusion

# In this project, I successfully implemented an architecture for diagonal integration of two modalities with the aim of achieving cohesive cross-generation between these modalities. The architecture is composed of a two-channel neural network. Each channel is processed through a channel-specific encoder before being passed into a shared autoencoder. The output from the shared autoencoder is then processed through a channel-specific decoder.
# 
# This architecture has been implemented using PyTorch, and the model’s training process utilized the Adam optimizer and the Mean Squared Error (MSE) as the loss function. To enhance the model’s performance and prevent overfitting, I incorporated several techniques like L1 regularization, a learning rate scheduler, and early stopping to see their effects on the model's performance.The initial high MSE in the first epoch could be due to the model starting with random weights, and thus making poor predictions. As training progresses, the model learns from the data, and the MSE decreases significantly. 
# 
# The training process after hyperparameter tuning seems to show a consistent decrease in the MSE over epochs, indicating that the model was effectively learning from the data. However, the final performance was not satisfactory and could benefit more by adjusting the model's architecture, tuning hyperparameters, or preprocessing the data differently.
# 
