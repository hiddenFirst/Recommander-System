import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ------ data load and processing ------

data_path = 'combined_Dataset.csv'
data = pd.read_csv(data_path) 

#separated the feature and label 
user_features = data[['User Caloric Value', 'User Fat Limit', 'User Carbohydrates Limit', 
                      'User Sugar Limit', 'User Protein Min', 'User Dietary Fiber Min', 
                      'User Cholesterol Limit', 'User Sodium Limit']]

food_feature = data.drop(columns=['User Caloric Value', 'User Fat Limit', 'User Carbohydrates Limit', 
                      'User Sugar Limit', 'User Protein Min', 'User Dietary Fiber Min', 
                      'User Cholesterol Limit', 'User Sodium Limit', 'Match Label'])

labels = data['Match Label']

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
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64,32)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)
        return x

class ItemTower(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super(ItemTower, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64,32)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)
        return x

# ------ define two-tower model -------

class TwoTowerModel(nn.Module):
    def __init__(self, user_dimension, item_dimension):
        super(TwoTowerModel, self).__init__()
        self.user_tower = UserTower(user_dimension)
        self.item_Tower = ItemTower(item_dimension)
    def forward(self, user_input, item_input):
        # using dot product to get score - higher dot product = higher score

        user = self.user_tower(user_input)
        item = self.item_tower(item_input)

        dot_product = torch.sum(user * item, dim=-1)
        return dot_product
