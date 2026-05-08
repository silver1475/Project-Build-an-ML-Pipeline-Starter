import os
import argparse
import wandb
import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def go(args):
    # Initialize W&B
    run = wandb.init(job_type="train_random_forest")
    run.config.update(vars(args))

    # --- 1. Robust Data Loading ---
    project_root = os.getcwd()
    local_candidates = [
        os.path.join(project_root, "trainval_data.csv"),
        os.path.join(project_root, "data", "trainval_data.csv"),
    ]
    
    local_trainval = next((path for path in local_candidates if os.path.exists(path)), None)

    if local_trainval is None:
        try:
            artifact = run.use_artifact(args.trainval_artifact)
            local_trainval = artifact.file()
        except Exception:
            raise FileNotFoundError(f"Could not find {args.trainval_artifact} locally or on W&B")

    df = pd.read_csv(local_trainval)

    # --- 2. Data Cleaning (Crucial for TfidfVectorizer) ---
    # TfidfVectorizer crashes on NaN values. We fill them with empty strings.
    if "name" in df.columns:
        df["name"] = df["name"].fillna("").astype(str)

    # Get numeric columns
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'price' in numeric_features:
        numeric_features.remove('price')

    # Optional: Handle numeric NaNs if any exist (filling with median is usually safer than 0)
    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())
    
    # Identify if the text column 'name' is present for feature selection
    text_feature = "name" if "name" in df.columns else None
    feature_cols = numeric_features + ([text_feature] if text_feature else [])
    
    X = df[feature_cols]
    y = df["price"]

    # --- 3. Splitting & Stratification ---
    stratify_col = None
    if args.stratify_by != "none":
        if args.stratify_by in df.columns:
            stratify_col = df[args.stratify_by]
        else:
            print(f"Warning: Stratification column '{args.stratify_by}' not found.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=args.val_size, 
        random_state=args.random_seed,
        stratify=stratify_col
    )

    # --- 4. Build Dynamic Pipeline ---
    with open(args.rf_config, "r") as f:
        rf_params = json.load(f)

    # Overwrite/set random_state in the dict to ensure consistency
    rf_params["random_state"] = args.random_seed

    transformers = []
    if text_feature:
        transformers.append(
            ('text', TfidfVectorizer(max_features=args.max_tfidf_features, stop_words='english'), text_feature)
        )
    if numeric_features:
        transformers.append(('num', 'passthrough', numeric_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('rf', RandomForestRegressor(**rf_params))
    ])

    # --- 5. Train ---
    print(f"Training RF with {len(numeric_features)} numeric features and text='{text_feature}'")
    pipe.fit(X_train, y_train)

    # --- 6. Export and Log Artifact ---
    export_dir = os.path.join(project_root, args.output_artifact)
    os.makedirs(export_dir, exist_ok=True)
    model_path = os.path.join(export_dir, "model.joblib")
    
    joblib.dump(pipe, model_path)

    artifact = wandb.Artifact(
        args.output_artifact,
        type="model_export",
        description="Trained Random Forest pipeline"
    )
    artifact.add_dir(export_dir)
    run.log_artifact(artifact)
    
    print(f"Success! Model exported to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Random Forest model")
    
    parser.add_argument("--trainval_artifact", type=str, required=True)
    parser.add_argument("--val_size", type=float, required=True)
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--stratify_by", type=str, required=True)
    parser.add_argument("--rf_config", type=str, required=True)
    parser.add_argument("--max_tfidf_features", type=int, required=True)
    parser.add_argument("--output_artifact", type=str, required=True)
    
    args = parser.parse_args()
    go(args)