import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

try:
    # pyrefly: ignore [missing-import]
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder = None
    print("Warning: category_encoders not installed. Target Encoding will be skipped.")

def main():
    print("Loading Datasets:")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'train.csv')

    if not os.path.exists(file_path):
        print(f" Error cannot find '{file_path}'")
        return

    df = pd.read_csv(file_path)
    print(f"Dataset Loaded Successfully. Rows:{df.shape[0]}, Features:{df.shape[1]}\n")

    # ── 1. HANDLING MISSING DATA ─────────────────────────────────────────────
    print("HANDLING MISSING DATA")
    print("Artificially deleting some 'Age' data to demonstrate imputation for the lesson.")

    df.loc[0:25, 'Age'] = np.nan
    imputer = SimpleImputer(strategy="median")
    df['Age'] = imputer.fit_transform(df[['Age']])
    print(f"Imputation complete. 'Age' now has {df['Age'].isnull().sum()} null values.\n")

    # ── 2. LOG TRANSFORMATION ─────────────────────────────────────────────────
    print("Evaluating the skewness of the Spending distribution...")
    df["LogSpending"] = np.log1p(df['Spending'])
    print(f"Log Transformation applied. New skewness: {df['LogSpending'].skew():.2f} (closer to 0 is perfectly balanced).\n")

    # ── 3. TARGET ENCODING ────────────────────────────────────────────────────
    if TargetEncoder is not None:
        print("Applying Target Encoder on 'City' (categorical) with 'Visits_Per_Month' as target.")
        encoder = TargetEncoder()
        # TargetEncoder requires no missing values in the target column
        df['Visits_Per_Month'] = SimpleImputer(strategy="median").fit_transform(df[['Visits_Per_Month']])
        df["City_Encoded"] = encoder.fit_transform(df["City"], df['Visits_Per_Month'])
        print(f"Target Encoding complete. Sample:\n{df[['City','City_Encoded']].drop_duplicates().head()}\n")
    else:
        print("Category encoders not installed. Skipping Target Encoding.\n")

    # ── 4. FEATURE SELECTION ──────────────────────────────────────────────────
    print("FEATURE SELECTION")
    features_to_rank = ["Age", "Spending", "Visits_Per_Month"]
    X_features = df[features_to_rank].copy()
    X_features = SimpleImputer(strategy="median").fit_transform(X_features)
    y_target = df["Visits_Per_Month"].fillna(df["Visits_Per_Month"].median())

    selector = SelectKBest(score_func=mutual_info_regression, k=2)
    selector.fit(X_features, y_target)
    winning_features = selector.get_support()
    best_features = [features_to_rank[i] for i, keep in enumerate(winning_features) if keep]
    print(f"Top 2 features selected by SelectKBest: {best_features}\n")

    # ── 5. TRAIN / TEST SPLIT ─────────────────────────────────────────────────
    print("SPLITTING DATA")
    X = df[best_features].fillna(df[best_features].median())
    Y = df["Spending"].fillna(df["Spending"].median())   # predict Spending
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    print(f"Training data size: {X_train.shape}")
    print(f"Testing data size:  {X_test.shape}\n")

    # ── 6. LINEAR REGRESSION ──────────────────────────────────────────────────
    print("TRAINING LINEAR REGRESSION MODEL")
    model = LinearRegression()
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    score = model.score(X_test, Y_test)
    print(f"Model R² score on test set: {score:.4f}\n")

    # ── 7. PREDICTION vs ACTUAL COMPARISON ───────────────────────────────────
    print("SAMPLE PREDICTIONS vs ACTUAL:")
    actual_spending = Y_test.head(3).values
    predicted_spending = predictions[:3]

    for i in range(3):
        predicted = round(predicted_spending[i])
        actual = actual_spending[i]
        difference = abs(actual - predicted)
        print(f"  Model Guessed : {predicted}")
        print(f"  Real Answer   : {actual}")
        print(f"  Difference    : {difference}\n")

if __name__ == "__main__":
    main()
