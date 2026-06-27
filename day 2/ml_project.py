import pandas as pd
import numpy as np 
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest,mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os


try:
    # pyrefly: ignore [missing-import]
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder=None
    print("Warning:category_encoders not installed.Target Encoding will ")

def main():

    print("Loading Datasets:")
    # Look for train.csv in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'train.csv')

    if not os.path.exists(file_path):
        print(f" Error cannot find '{file_path}'")
        return 
    df=pd.read_csv(file_path)
    print(f"Dataset Loaded Succesfully.Rows:{df.shape[0]},Features:{df.shape[1]}\n")


if __name__=="__main__":
    main()
    