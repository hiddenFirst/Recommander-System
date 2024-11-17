import torch
import joblib
import pandas as pd
import numpy as np
from recommender import TwoTowerModel

#load trained model and scalers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TwoTowerModel(user_dimension=9, item_dimension=9)
model.load_state_dict(torch.load('trained_model.pth', map_location=device, weights_only=True))
model.eval()
scaler_user = joblib.load('scaler_user.pkl')
scaler_food = joblib.load('scaler_food.pkl')

#load and scale food data
food_data = pd.read_csv('FOOD-DATA-GROUP-Prossced.csv')
food_features = food_data.drop(columns=['food'])
food_features_scaled = scaler_food.transform(food_features)

#get user input
def get_user_input():
    print("Please enter your nutritional needs in the following format (with no spaces after commas):")
    print("User ID, User Caloric Value, User Fat Limit, User Carbohydrates Limit, User Sugar Limit, User Protein Min, User Dietary Fiber Min, User Cholesterol Limit, User Sodium Limit")
    user_input_str = input("Paste your input here: ")
    user_input_list = list(map(float, user_input_str.split(',')))
    if len(user_input_list) != 9:
        raise ValueError("Input must contain exactly 9 values.")

    user_data = {
        'User ID': int(user_input_list[0]),
        'User Caloric Value': user_input_list[1],
        'User Fat Limit': user_input_list[2],
        'User Carbohydrates Limit': user_input_list[3],
        'User Sugar Limit': user_input_list[4],
        'User Protein Min': user_input_list[5],
        'User Dietary Fiber Min': user_input_list[6],
        'User Cholesterol Limit': user_input_list[7],
        'User Sodium Limit': user_input_list[8]
    }
    return pd.DataFrame([user_data])

#recommend foods
def recommend_foods(user_input_scaled):
    user_input_tensor = torch.tensor(user_input_scaled, dtype=torch.float32).to(device)
    food_features_tensor = torch.tensor(food_features_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        scores = model(user_input_tensor, food_features_tensor)
        scores = scores.cpu().numpy().flatten()
    
    print("Scores for foods:", scores)  #check if scores are reasonable

    recommended_indices = np.argsort(scores)[::-1]  #sort in descending order
    recommended_foods = food_data.iloc[recommended_indices[:10]]  #top 10 recommendations only
    return recommended_foods

def main():
    user_input = get_user_input()
    user_input_scaled = scaler_user.transform(user_input)
    recommendations = recommend_foods(user_input_scaled)
    
    print("\nRecommended Foods:")
    print(recommendations[['food']]) 

if __name__ == "__main__":
    main()
