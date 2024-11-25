# Recommander-System-
This project implements a deep learning-based two-tower recommendation system. The model takes user nutritional requirements as input and recommends the most suitable food items. The system is built with PyTorch and trained using paired user-food data.

## Obtaining Dataset
Unzip the datset in same directory to get a `combined_Dataset.csv`
- Do not push the csv dataset into the remote repository (size limit exceeded)
- For local reference only

## Project Structure
```
├── recommander.py           # Model training script
├── interact.py              # Interactive recommendation script
├── FOOD-DATA-GROUP-Prossced.csv  # Original food dataset
├── combined_Dataset.csv      # User-food paired dataset (for training)
├── trained_model.pth         # Trained model
├── scaler_user.pkl           # Standard scaler for user features
├── scaler_food.pkl           # Standard scaler for food features
├── README.md                 # Project description
```


## Features
1. Model Training:

- recommander.py defines the two-tower model (TwoTowerModel), which generates embeddings for user and food features.
- Uses combined_Dataset.csv to train the model.
- Saves the trained model and scalers upon completion.

2. Recommendation Functionality:

- Interact.py provides an interactive command-line tool where users can input their nutritional requirements to receive food recommendations.
- Recommendations are generated using FOOD-DATA-GROUP-Prossced.csv.

3. Data Processing:

- User and food features are standardized using StandardScaler.
- Combined_Dataset.csv contains paired user-food data with labels (indicating whether the food meets the user's requirements).

## Requirements
Install the following dependencies:

- Python 3.8+
- PyTorch
- scikit-learn
- pandas
- numpy
- joblib

## Datasets
1. Original Food Dataset (FOOD-DATA-GROUP-Prossced.csv)
   This dataset contains nutritional features for each food item, including:
   - Caloric Value
   - Fat
   - Carbohydrates
   - Sugars
   - Protein
   - Dietary Fiber
   - Cholesterol
   - Sodium
2. Paired Dataset (combined_Dataset.csv)
   This dataset pairs user features with food features and includes a label (Label) indicating whether the food meets the user's nutritional requirements:

   - User features (e.g., caloric need, minimum protein requirement, etc.)
   - Food features (e.g., calories, fat, sugar, etc.)
   - Label (1 for a match, 0 for a mismatch)

## How to Use
1. Train the Model
   Run recommander.py to train the model and save the trained parameters:
   ```
   ├── python recommander.py
   ```
   
   After training, the following files will be generated:

   ```trained_model.pth```: Saved model parameters.
   ```scaler_user.pkl```: User feature scaler.
   ```scaler_food.pkl```: Food feature scaler.
   
3. Use the Recommendation System
   Run interact.py for an interactive recommendation system:
   ```
   ├── python interact.py
   ```
   Provide your nutritional requirements in the following format (comma-separated):
   ```
   Enter the following format:
   User ID, User Caloric Value, User Fat Limit, User Carbohydrates Limit, User Sugar Limit, 
   User Protein Min, User Dietary Fiber Min, User Cholesterol Limit, User Sodium Limit

   Example input:
   1, 2000, 70, 300, 50, 60, 30, 200, 1500
   ```
