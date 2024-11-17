import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import joblib

# ------ data load and processing ------

data_path = 'combined_Dataset.csv'
data = pd.read_csv(data_path) 

#separated the feature and label 
user_features = data[['User ID', 'User Caloric Value', 'User Fat Limit', 'User Carbohydrates Limit', 
                      'User Sugar Limit', 'User Protein Min', 'User Dietary Fiber Min', 
                      'User Cholesterol Limit', 'User Sodium Limit']]

food_feature = data.drop(columns=['food', 'User ID', 'User Caloric Value', 'User Fat Limit', 'User Carbohydrates Limit', 
                      'User Sugar Limit', 'User Protein Min', 'User Dietary Fiber Min', 
                      'User Cholesterol Limit', 'User Sodium Limit', 'Label'])

labels = data['Label']

#data standerlized 
scaler_user = StandardScaler()
scaler_food = StandardScaler()
user_features_scaled = scaler_user.fit_transform(user_features) 
food_features_scaled = scaler_food.fit_transform(food_feature)

#split to train and val dataset 
X_user_train, X_user_val, X_food_train, X_food_val, y_train, y_val = train_test_split(
    user_features_scaled, food_features_scaled, labels, test_size=0.2, random_state=42
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation will be performed on: {device}")

# convert to PyTorch tensor 
X_user_train_tensor = torch.tensor(X_user_train, dtype=torch.float32).to(device)
X_user_val_tensor = torch.tensor(X_user_val, dtype=torch.float32).to(device) 
X_food_train_tensor = torch.tensor(X_food_train, dtype=torch.float32).to(device)
X_food_val_tensor = torch.tensor(X_food_val, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1).to(device)

#to create dataloader 
batch_size = 64
train_dataset = TensorDataset(X_user_train_tensor, X_food_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_user_val_tensor, X_food_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# ------ define individual towers -------
# each tower is initially defined very simple: with two fully connected layers followed by a RELU activation function

class UserTower(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super(UserTower, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.model(x)

class ItemTower(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super(ItemTower, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)

# ------ define two-tower model -------

class TwoTowerModel(nn.Module):
    def __init__(self, user_dimension, item_dimension):
        super(TwoTowerModel, self).__init__()
        self.user_tower = UserTower(user_dimension)
        self.item_tower = ItemTower(item_dimension)
    def forward(self, user_input, item_input):
        # using dot product to get score - higher dot product = higher score

        user = self.user_tower(user_input)
        item = self.item_tower(item_input)

        dot_product = torch.sum(user * item, dim=-1)
        return dot_product

if __name__ == "__main__":
    # ------- Model training and validation --------- 

    # Initialize the model, loss function, and optimizer 
    user_input_dim = X_user_train.shape[1] 
    food_input_dim = X_food_train.shape[1] 
    model = TwoTowerModel(user_input_dim, food_input_dim).to(device) 
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.0001) 

    # Training model 
    epochs = 20 
    for epoch in range(epochs): 
        model.train() 
        train_loss = 0 
        correct_train = 0
        total_train = 0

        for user_input, food_input, label in train_loader: 
            user_input = user_input.to(device)
            food_input = food_input.to(device)
            label = label.to(device)

            optimizer.zero_grad() 
            scores = model(user_input, food_input)
            scores = scores.view(-1, 1)
            loss = criterion(scores, label) 
            loss.backward() 
            optimizer.step() 

            train_loss += loss.item()  
            predictions = torch.sigmoid(scores) 
            predicted_labels = (predictions > 0.5).float()
            correct_train += (predicted_labels == label).sum().item()
            total_train += label.size(0)
        train_accuracy = correct_train / total_train

    # Val model 
        model.eval() 
        val_loss = 0 
        correct_val = 0
        total_val = 0
        with torch.no_grad(): 
            for user_input, food_input, label in val_loader: 
                user_input = user_input.to(device)
                food_input = food_input.to(device)
                label = label.to(device)
            
                scores = model(user_input, food_input) 
                scores = scores.view(-1, 1)
                loss = criterion(scores, label) 
                val_loss += loss.item() 

                predictions = torch.sigmoid(scores)
                predicted_labels = (predictions > 0.5).float()
                correct_val += (predicted_labels == label).sum().item()
                total_val += label.size(0)
        val_accuracy = correct_val / total_val

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Training Loss: {train_loss/len(train_loader):.4f}, "
              f"Training Accuracy: {train_accuracy:.4f}, "
              f"Validation Loss: {val_loss/len(val_loader):.4f}, "
              f"Validation Accuracy: {val_accuracy:.4f}")



    torch.save(model.state_dict(), 'trained_model.pth')
    print("model save to 'trained_model.pth'")

    joblib.dump(scaler_user, 'scaler_user.pkl')
    joblib.dump(scaler_food, 'scaler_food.pkl')
    print("StandardScaler save to 'scaler_user.pkl' and 'scaler_food.pkl'")

