#!/usr/bin/env python3
"""
Self-Supervised Distillation (SSD) Pipeline for Self-Diffusion Coefficient Prediction

This script implements the iterative self-supervised distillation approach:
1. Train a teacher model on experimental data
2. Use the teacher to predict on MD simulation data
3. Filter predictions by agreement threshold
4. Augment training data with filtered MD predictions
5. Train a new student model on augmented data
6. Repeat for multiple cycles

Usage:
    cd scripts/
    python run_ssd.py --threshold 1.0 --cycles 4 --output_dir ../data/SelfDiff_SSD_output

Requirements:
    - pandas, numpy, scikit-learn, joblib, kneed
    - transformers (for RoBERTa embeddings)
    - pyarrow (for parquet storage)
"""

import argparse
import gc
import glob
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from kneed import KneeLocator
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import (
    DataCollatorWithPadding,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
)


# RoBERTa Embedding Functions

def load_roberta_model():
    """Load RoBERTa tokenizer and model for SMILES embeddings."""
    tokenizer = RobertaTokenizerFast.from_pretrained(
        "entropy/roberta_zinc_480m", max_len=128
    )
    model = RobertaForMaskedLM.from_pretrained("entropy/roberta_zinc_480m")
    collator = DataCollatorWithPadding(tokenizer, padding=True, return_tensors="pt")
    return tokenizer, model, collator


def create_embedding(smiles_list, tokenizer, model, collator):
    """Create RoBERTa embeddings for a list of SMILES strings."""
    inputs = collator(tokenizer(smiles_list))
    outputs = model(**inputs, output_hidden_states=True)
    full_embeddings = outputs[1][-1]
    mask = inputs["attention_mask"]
    embeddings = (full_embeddings * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)
    return embeddings


def batch_embed_smiles(smiles_list, output_dir, cycle, tokenizer, model, collator,
                       batch_size=16, prefix="df"):
    """
    Embed SMILES in batches, saving to parquet files for memory efficiency.
    Returns a dictionary mapping SMILES to embedding vectors.
    """
    unique_smiles = list(set(s for s in smiles_list if pd.notna(s)))
    
    out_path = Path(output_dir) / f"embedded_SMILES_parts_{prefix}_cycle_{cycle}"
    out_path.mkdir(exist_ok=True)
    
    # Check already processed
    done = set()
    for p in out_path.glob("part_*.parquet"):
        tbl = pq.read_table(p, columns=["can_SMILES"])
        done.update(tbl.column("can_SMILES").to_pylist())
    
    todo = [s for s in unique_smiles if s not in done]
    print(f"Total: {len(unique_smiles)} | already saved: {len(done)} | remaining: {len(todo)}")
    
    emb_cols = None
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        emb = create_embedding(batch, tokenizer, model, collator).detach().cpu().numpy()
        
        if emb_cols is None:
            emb_cols = [f"e{j}" for j in range(emb.shape[1])]
        
        df_emb = pd.DataFrame(emb, columns=emb_cols)
        df_emb.insert(0, "can_SMILES", batch)
        
        ts = int(time.time() * 1000)
        part_path = out_path / f"part_{ts}_{len(batch)}.parquet"
        df_emb.to_parquet(part_path, index=False, compression="zstd")
        
        del emb
        gc.collect()
        
        if (i // batch_size) % 10 == 0:
            print(f"Progress: {i + len(batch)}/{len(todo)} newly saved")
    
    # Load all embeddings
    parts = sorted(glob.glob(str(out_path / "part_*.parquet")))
    if not parts:
        raise FileNotFoundError("No parquet parts found.")
    
    tables = [pq.read_table(p) for p in parts]
    full = pa.concat_tables(tables)
    df_emb_all = full.to_pandas()
    
    embedded_smiles = {
        s: v for s, v in zip(
            df_emb_all["can_SMILES"].tolist(),
            df_emb_all.iloc[:, 1:].to_numpy()
        )
    }
    print(f"Loaded {len(embedded_smiles)} embeddings")
    return embedded_smiles


def prepare_features(df, embedded_smiles, pca_model=None, fit_pca=False):
    """
    Prepare feature matrix from SMILES embeddings and temperature.
    
    Args:
        df: DataFrame with 'can_SMILES' and 'T_K' columns
        embedded_smiles: dict mapping SMILES to embedding vectors
        pca_model: fitted PCA model (or None to fit new one)
        fit_pca: whether to fit PCA on this data
    
    Returns:
        combined_embeddings, pca_model, embeddings_reduced
    """
    embeddings = []
    for _, row in df.iterrows():
        embeddings.append(list(embedded_smiles[row["can_SMILES"]]) + [row["T_K"]])
    embeddings = np.array(embeddings)
    
    embeddings_roberta = embeddings[:, :-1]
    temperature_feature = embeddings[:, -1].reshape(-1, 1)
    
    if fit_pca:
        # Determine optimal PCA components using knee detection
        pca_full = PCA(n_components=min(embeddings_roberta.shape))
        pca_full.fit(embeddings_roberta)
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
        components = np.arange(1, len(cumulative_variance) + 1)
        
        knee_locator = KneeLocator(
            components, cumulative_variance, curve="concave", direction="increasing"
        )
        knee_point = knee_locator.knee
        print(f"Optimal PCA components (knee point): {knee_point}")
        
        pca_model = PCA(n_components=int(knee_point))
        embeddings_reduced = pca_model.fit_transform(embeddings_roberta)
    else:
        embeddings_reduced = pca_model.transform(embeddings_roberta)
    
    combined_embeddings = np.concatenate([embeddings_reduced, temperature_feature], axis=1)
    print(f"Combined features shape: {combined_embeddings.shape}")
    
    return combined_embeddings, pca_model, embeddings_reduced


# Model Training Functions

def train_rf_model(X, y, output_dir, cycle, verbose=2):
    """
    Train Random Forest model with grid search and cross-validation.
    
    Returns:
        best_model, metrics_dict, grid_search results
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(random_state=1, criterion="absolute_error"))
    ])
    
    param_grid = {
        "rf__n_estimators": [10, 50, 100, 200],
        "rf__max_depth": [2, 5, 10],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf": [1, 2, 4]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=verbose)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_pred_all = best_model.predict(X)
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    # Metrics
    metrics = {
        "R2_all": r2_score(y, y_pred_all),
        "MAE_all": mean_absolute_error(y, y_pred_all),
        "R2_train": r2_score(y_train, y_pred_train),
        "MAE_train": mean_absolute_error(y_train, y_pred_train),
        "R2_test": r2_score(y_test, y_pred_test),
        "MAE_test": mean_absolute_error(y_test, y_pred_test),
    }
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Test R²: {metrics['R2_test']:.4f}, Test MAE: {metrics['MAE_test']:.4f}")
    
    # Cross-validation for prediction variance
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    all_preds = np.zeros((X.shape[0], 5))
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        model_cv = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(
                random_state=1,
                criterion="absolute_error",
                **{k.replace("rf__", ""): v for k, v in grid_search.best_params_.items()}
            ))
        ])
        model_cv.fit(X[train_idx], y.iloc[train_idx])
        all_preds[:, fold] = model_cv.predict(X)
    
    cv_results = {
        "y_pred": y_pred_all,
        "pred_cv_mean": np.mean(all_preds, axis=1),
        "pred_cv_std": np.std(all_preds, axis=1),
        "y_train": y_train,
        "y_test": y_test,
        "X_train": X_train,
        "X_test": X_test,
    }
    
    return best_model, metrics, grid_search, cv_results


def save_cycle_results(df, y, metrics, grid_search, cv_results, output_dir, cycle):
    """Save all results for a training cycle."""
    # Add predictions to dataframe
    df = df.copy()
    df["Pred"] = cv_results["y_pred"]
    df["Pred_CV_Mean"] = cv_results["pred_cv_mean"]
    df["Pred_CV_Std"] = cv_results["pred_cv_std"]
    df["y_true"] = y.values
    df["is_train"] = df.index.isin(cv_results["y_train"].index)
    df["is_test"] = df.index.isin(cv_results["y_test"].index)
    
    # Save results
    df.to_csv(os.path.join(output_dir, f"SelfDiff_cycle_{cycle}_results.csv"), index=False)
    
    # Save test performance
    test_perf = pd.DataFrame({
        "cycle": [cycle],
        "R2_test": [metrics["R2_test"]],
        "MAE_test": [metrics["MAE_test"]],
        "n_test": [len(cv_results["y_test"])]
    })
    test_perf.to_csv(os.path.join(output_dir, f"SelfDiff_cycle_{cycle}_test_performance.csv"), index=False)
    
    # Save CV results
    cv_df = pd.DataFrame(grid_search.cv_results_)
    cv_df.to_csv(os.path.join(output_dir, f"SelfDiff_RT_rf_roberta_cv_results_cycle_{cycle}.csv"), index=False)
    
    # Save best params and metrics
    with open(os.path.join(output_dir, f"SelfDiff_RT_rf_roberta_best_params_cycle_{cycle}.txt"), "w") as f:
        f.write(f"Cycle: {cycle}\n")
        f.write(f"Best Parameters:\n{grid_search.best_params_}\n\n")
        f.write(f"Mean CV score: {grid_search.cv_results_['mean_test_score'][grid_search.best_index_]}\n")
        f.write(f"Std CV score: {grid_search.cv_results_['std_test_score'][grid_search.best_index_]}\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
        f.write(f"n_train: {len(cv_results['y_train'])}\n")
        f.write(f"n_test: {len(cv_results['y_test'])}\n")
    
    return df


# SSD Pipeline Functions

def run_teacher_cycle(output_dir, tokenizer, model, collator):
    """
    Run the teacher (experimental) cycle.
    Train initial model on experimental self-diffusion data.
    """
    print("\n" + "="*60)
    print("CYCLE: exp (Teacher Model)")
    print("="*60)
    
    cycle = "exp"
    
    # Load experimental data (295-300K)
    df = pd.read_csv("../data/processed/SelfDiff_exp_processed.csv")
    df = df[(df.T_K <= 300) & (df.T_K >= 295)]
    df.dropna(subset="can_SMILES", inplace=True)
    
    # Remove duplicates by SMILES+temperature
    df["can_SMILES_T_K"] = df.apply(lambda r: f"{r.can_SMILES}@{r.T_K}", axis=1)
    df.drop_duplicates(subset="can_SMILES_T_K", inplace=True)
    df.to_csv(os.path.join(output_dir, f"SelfDiff_cycle_{cycle}.csv"), index=False)
    
    print(f"Experimental data: {len(df)} samples")
    
    # Embed SMILES
    embedded_smiles = batch_embed_smiles(
        df["can_SMILES"].tolist(), output_dir, cycle,
        tokenizer, model, collator, prefix="df"
    )
    
    # Prepare features with PCA
    combined_embeddings, pca_model, embeddings_reduced = prepare_features(
        df, embedded_smiles, fit_pca=True
    )
    
    # Save PCA model and knee point
    with open(os.path.join(output_dir, f"pca_object_cycle_{cycle}.pkl"), "wb") as f:
        pickle.dump(pca_model, f)
    
    with open(os.path.join(output_dir, f"embeddings_roberta_reduced_cycle_{cycle}.pkl"), "wb") as f:
        pickle.dump({"embeddings": embeddings_reduced, "cycle": cycle}, f)
    
    # Train model
    y = df["Dexp×10-9/m2·s-1"]
    best_model, metrics, grid_search, cv_results = train_rf_model(
        combined_embeddings, y, output_dir, cycle
    )
    
    # Save results
    save_cycle_results(df, y, metrics, grid_search, cv_results, output_dir, cycle)
    joblib.dump(best_model, os.path.join(output_dir, f"SelfDiff_RT_rf_roberta_best_cycle_{cycle}.joblib"))
    
    return pca_model


def run_student_cycle(cycle, prev_cycle, threshold, output_dir, pca_model,
                      tokenizer, model, collator, is_first_student=False):
    """
    Run a student cycle.
    Use previous model to predict on MD data, filter by threshold, augment training data.
    """
    print("\n" + "="*60)
    print(f"CYCLE: {cycle} (Student Model)")
    print("="*60)
    
    # Load previous cycle data
    df_prev = pd.read_csv(os.path.join(output_dir, f"SelfDiff_cycle_{prev_cycle}.csv"))
    
    if is_first_student:
        # First student: load MD simulation data
        df_mol = pd.read_csv("../data/processed/SelfDiff_MDmolecules.csv")
        id2smiles = df_mol.set_index("MD_ID")["can_SMILES"].to_dict()
        
        df_add = pd.read_csv("../data/MD/selfdiff/gaff2/diffusion_coefficients_adaptive_full.csv")
        df_add["can_SMILES"] = df_add["molecule"].map(id2smiles)
        df_add["T_K"] = 300
        df_add = df_add.rename(columns={"molecule": "MD_ID", "D_m2_s": "Dmd_m2_s"})
        df_add["Dmd×10-9/m2·s-1"] = df_add["Dmd_m2_s"] * 1e9
    else:
        # Subsequent students: use leftover from previous cycle
        df_add = pd.read_csv(os.path.join(output_dir, f"SelfDiff_cycle_{prev_cycle}_leftover.csv"))
    
    # Remove molecules already in training set
    df_add = df_add[~df_add["can_SMILES"].isin(df_prev["can_SMILES"])]
    print(f"Previous data: {len(df_prev)}, New candidates: {len(df_add)}")
    
    if len(df_add) == 0:
        print("No new molecules to add. Stopping.")
        return None
    
    # Embed new SMILES
    embedded_smiles = batch_embed_smiles(
        df_add["can_SMILES"].tolist(), output_dir, cycle,
        tokenizer, model, collator, prefix="df_add"
    )
    
    # Prepare features using existing PCA
    combined_embeddings_add, _, _ = prepare_features(
        df_add, embedded_smiles, pca_model=pca_model, fit_pca=False
    )
    
    # Load previous model and predict
    prev_model = joblib.load(os.path.join(output_dir, f"SelfDiff_RT_rf_roberta_best_cycle_{prev_cycle}.joblib"))
    y_pred = prev_model.predict(combined_embeddings_add)
    df_add["Dpred×10-9/m2·s-1"] = y_pred
    
    # Filter by threshold
    agreement = abs(df_add["Dmd×10-9/m2·s-1"] - df_add["Dpred×10-9/m2·s-1"])
    df_aug = df_add[agreement < threshold].copy()
    df_leftover = df_add[agreement >= threshold].copy()
    
    df_aug["ln(Dmd×10-9/m2·s-1)"] = np.log(df_aug["Dmd×10-9/m2·s-1"])
    df_aug["ln(Dpred×10-9/m2·s-1)"] = np.log(df_aug["Dpred×10-9/m2·s-1"])
    df_aug["ref"] = "MD"
    
    print(f"Augmented (threshold < {threshold}): {len(df_aug)}")
    print(f"Leftover: {len(df_leftover)}")
    
    df_aug.to_csv(os.path.join(output_dir, f"SelfDiff_cycle_{cycle}_aug.csv"), index=False)
    df_leftover.to_csv(os.path.join(output_dir, f"SelfDiff_cycle_{cycle}_leftover.csv"), index=False)
    
    # Combine with previous data
    df = pd.concat([df_prev, df_aug])
    cols = ["can_SMILES", "T_K", "Dexp×10-9/m2·s-1", "box", "window_ps",
            "Dmd×10-9/m2·s-1", "Dpred×10-9/m2·s-1", "ref"]
    df = df[[c for c in cols if c in df.columns]]
    
    # Create target: prefer experimental, else use MD prediction
    df["target"] = df["Dexp×10-9/m2·s-1"].fillna(df["Dpred×10-9/m2·s-1"])
    
    # Sanity check
    both_present = df["Dexp×10-9/m2·s-1"].notna() & df["Dpred×10-9/m2·s-1"].notna()
    assert not both_present.any(), "Found rows with both Dexp and Dpred!"
    
    df.to_csv(os.path.join(output_dir, f"SelfDiff_cycle_{cycle}.csv"), index=False)
    
    # Embed all SMILES in combined dataset
    embedded_smiles = batch_embed_smiles(
        df["can_SMILES"].tolist(), output_dir, cycle,
        tokenizer, model, collator, prefix="df"
    )
    
    # Prepare features
    combined_embeddings, _, embeddings_reduced = prepare_features(
        df, embedded_smiles, pca_model=pca_model, fit_pca=False
    )
    
    # Save reduced embeddings
    with open(os.path.join(output_dir, f"embeddings_roberta_reduced_cycle_{cycle}.pkl"), "wb") as f:
        pickle.dump({"embeddings": embeddings_reduced, "cycle": cycle}, f)
    
    # Train model
    y = df["target"]
    best_model, metrics, grid_search, cv_results = train_rf_model(
        combined_embeddings, y, output_dir, cycle
    )
    
    # Save results
    save_cycle_results(df, y, metrics, grid_search, cv_results, output_dir, cycle)
    joblib.dump(best_model, os.path.join(output_dir, f"SelfDiff_RT_rf_roberta_best_cycle_{cycle}.joblib"))
    
    return len(df_leftover)


# Main

def main():
    parser = argparse.ArgumentParser(description="Run SSD pipeline for self-diffusion prediction")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Agreement threshold for filtering MD predictions (default: 1.0)")
    parser.add_argument("--cycles", type=int, default=4,
                        help="Number of student cycles to run (default: 4)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: ../data/SelfDiff_SSD_YYYYMMDD_thresholdX)")
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = f"../data/SelfDiff_SSD_{datetime.now().strftime('%Y%m%d')}_threshold{int(args.threshold)}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Cycles: {args.cycles}")
    
    # Load RoBERTa model
    print("\nLoading RoBERTa model...")
    tokenizer, model, collator = load_roberta_model()
    
    # Run teacher cycle
    pca_model = run_teacher_cycle(args.output_dir, tokenizer, model, collator)
    
    # Run student cycles
    prev_cycle = "exp"
    for i in range(1, args.cycles + 1):
        cycle = str(i)
        leftover = run_student_cycle(
            cycle, prev_cycle, args.threshold, args.output_dir, pca_model,
            tokenizer, model, collator, is_first_student=(i == 1)
        )
        
        if leftover is None or leftover == 0:
            print(f"\nStopping early: no more molecules to process after cycle {cycle}")
            break
        
        prev_cycle = cycle
    
    print("\n" + "="*60)
    print("SSD Pipeline Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
