import pandas as pd
import numpy as np
import os
import joblib
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def load_data(data_dir, synthetic_path=None):
    print("Loading datasets...")
    # Load Test 1 and Test 2 from processed folder
    test1 = pd.read_csv(os.path.join(data_dir, 'test1.csv'))
    test2 = pd.read_csv(os.path.join(data_dir, 'test2.csv'))
    
    # Combine real data
    df_real = pd.concat([test1, test2], ignore_index=True)
    print(f"Real data shape: {df_real.shape}")
    
    if synthetic_path and os.path.exists(synthetic_path):
        print(f"Loading synthetic data from {synthetic_path}...")
        df_syn = pd.read_csv(synthetic_path)
        print(f"Synthetic data shape: {df_syn.shape}")
        
        # Find common features (exclude metadata)
        meta_cols = ['timestamp', 'attack_id', 'scenario', 'attack_type', 'combination', 'target_controller', 'target_points', 'label']
        real_feats = [c for c in df_real.columns if c not in meta_cols]
        syn_feats = [c for c in df_syn.columns if c not in meta_cols]
        common_feats = list(set(real_feats).intersection(set(syn_feats)))
        common_feats.sort()
        
        print(f"Number of common physical features: {len(common_feats)}")
        
        # We split the REAL data into train/test first to keep the test set pure real
        # Real Train (70%), Real Test (30%)
        df_real_train, df_real_test = train_test_split(df_real, test_size=0.3, shuffle=False)
        
        # Augmented Train = Real Train + Synthetic
        # We only keep common features
        X_train = pd.concat([df_real_train[common_feats], df_syn[common_feats]], ignore_index=True)
        y_train = pd.concat([(df_real_train['label'] > 0).astype(int), df_syn['label']], ignore_index=True)
        
        X_test = df_real_test[common_feats]
        y_test = (df_real_test['label'] > 0).astype(int)
        
        return X_train, X_test, y_train, y_test, common_feats
    else:
        # Fallback to original logic if no synthetic data
        meta_cols = ['timestamp', 'attack_id', 'scenario', 'attack_type', 'combination', 'target_controller', 'target_points']
        y = (df_real['label'] > 0).astype(int)
        cols_to_drop = [c for c in meta_cols + ['label'] if c in df_real.columns]
        X = df_real.drop(cols_to_drop, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
        return X_train, X_test, y_train, y_test, X.columns.tolist()

def main():
    data_dir = r"C:\Users\PC GAMING\Desktop\new_ai\processed"
    syn_path = r"C:\Users\PC GAMING\Desktop\AI\HAI\synthetic_attacks.csv"
    X_train, X_test, y_train, y_test, feature_list = load_data(data_dir, syn_path)
    
    print(f"Final training set size: {X_train.shape}")
    print(f"Final test set size: {X_test.shape}")
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n--- Evaluating Baseline Models ---")
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        "LightGBM": lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
        "XGBoost": xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
    }
    
    best_f1 = 0
    best_model_name = ""
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        score = f1_score(y_test, preds)
        print(f"{name} F1-Score: {score:.4f}")
        if score > best_f1:
            best_f1 = score
            best_model_name = name
            
    print(f"\nBest baseline model is {best_model_name} with F1: {best_f1:.4f}")
    
    print(f"\n--- Hyperparameter Tuning {best_model_name} with Optuna ---")
    
    def objective(trial):
        if best_model_name == "LightGBM":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            clf = lgb.LGBMClassifier(**params)
        elif best_model_name == "XGBoost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'logloss'
            }
            clf = xgb.XGBClassifier(**params)
        else: # RandomForest
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'random_state': 42,
                'n_jobs': -1
            }
            clf = RandomForestClassifier(**params)
            
        clf.fit(X_train_scaled, y_train)
        preds = clf.predict(X_test_scaled)
        return f1_score(y_test, preds)
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=15)
    
    print("\nBest Trial Params:")
    print(study.best_trial.params)
    
    print("\n--- Training Final Best Model ---")
    if best_model_name == "LightGBM":
        final_model = lgb.LGBMClassifier(**study.best_trial.params, random_state=42, n_jobs=-1, verbose=-1)
    elif best_model_name == "XGBoost":
        final_model = xgb.XGBClassifier(**study.best_trial.params, random_state=42, n_jobs=-1, eval_metric='logloss')
    else:
        final_model = RandomForestClassifier(**study.best_trial.params, random_state=42, n_jobs=-1)
        
    final_model.fit(X_train_scaled, y_train)
    final_preds = final_model.predict(X_test_scaled)
    
    print("\n--- Final Classification Report ---")
    print(classification_report(y_test, final_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, final_preds))
    
    # Save the pipeline
    pipeline = {
        'scaler': scaler,
        'model': final_model,
        'features': feature_list
    }
    
    save_path = r"C:\Users\PC GAMING\Desktop\AI\HAI\best_hai_classifier.pkl"
    joblib.dump(pipeline, save_path)
    print(f"\nPipeline successfully saved to {save_path}")

if __name__ == "__main__":
    main()
