import pandas as pd
from train import train_model

def retrain():
    new_data = pd.read_csv("data/new_data.csv")
    train_model(new_data)

if __name__ == "__main__":
    retrain()
