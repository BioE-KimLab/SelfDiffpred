#!/usr/bin/env python3
"""
Control Experiment for Self-Supervised Distillation (SSD) Pipeline

This script implements a control experiment where instead of using the SSD
threshold-based filtering, we randomly sample the same number of molecules
that would have been selected by the SSD approach at each cycle.

This allows comparison between:
- SSD: Intelligent selection based on model agreement
- Control: Random selection with matched sample sizes

Usage:
    python run_ssd_control.py --ssd_dir ../data/SelfDiff_SSD_20251117_threshold1 \
                              --output_dir ../data/SelfDiff_SSD_20251117_threshold1_control

Requirements:
    - pandas, numpy, scikit-learn, joblib
    - transformers (for RoBERTa embeddings)
    - pyarrow (for parquet storage)
    - A completed SSD run (to get sample sizes per cycle)
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


def prepare_features(df, embedded_smiles, pca_model):
    """
    Prepare feature matrix from SMILES embeddings and temperature.
    Uses fit_transform (not just transform) for control experiment.
    """
    embeddings = []
    for _, row in df.iterrows():
        embeddings.append(list(embedded_smiles[row["can_SMILES"]]) + [row["T_K"]])
    embeddings = np.array(embeddings)
    
    embeddings_roberta = embeddings[:, :-1]
    temperature_feature = embeddings[:, -1].reshape(-1, 1)
    
    # Use fit_transform for control (different from SSD which uses transform)
    embeddings_reduced = pca_model.fit_transform(embeddings_roberta)
    
    combined_embeddings = np.concatenate([embeddings_reduced, temperature_feature], axis=1)
    print(f"Combined features shape: {combined_embeddings.shape}")
    
    return combined_embeddings, embeddings_reduced


# Model Training Functions

def train_rf_model(X, y, output_dir, cycle, verbose=2):
    """
    Train Random Forest model with grid search and cross-validation.
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
    }
    
    return best_model, metrics, grid_search, cv_results


def save_cycle_results(df, y, metrics, grid_search, cv_results, output_dir, cycle):
    """Save all results for a training cycle."""
    df = df.copy()
    df["Pred"] = cv_results["y_pred"]
    df["Pred_CV_Mean"] = cv_results["pred_cv_mean"]
    df["Pred_CV_Std"] = cv_results["pred_cv_std"]
    df["y_true"] = y.values
    df["is_train"] = df.index.isin(cv_results["y_train"].index)
    df["is_test"] = df.index.isin(cv_results["y_test"].index)
    
    df.to_csv(os.path.join(output_dir, f"SelfDiff_cycle_{cycle}_results.csv"), index=False)
    
    test_perf = pd.DataFrame({
        "cycle": [cycle],
        "R2_test": [metrics["R2_test"]],
        "MAE_test": [metrics["MAE_test"]],
        "n_test": [len(cv_results["y_test"])]
    })
    test_perf.to_csv(os.path.join(output_dir, f"SelfDiff_cycle_{cycle}_test_performance.csv"), index=False)
    
    cv_df = pd.DataFrame(grid_search.cv_results_)
    cv_df.to_csv(os.path.join(output_dir, f"SelfDiff_RT_rf_roberta_cv_results_cycle_{cycle}.csv"), index=False)
    
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


# Control Pipeline Functions

def get_ssd_sample_sizes(ssd_dir, cycles):
    """
    Get the number of samples added at each SSD cycle.
    This is used to match sample sizes in the control experiment.
    """
    sample_sizes = {}
    for i in range(1, cycles + 1):
        aug_file = os.path.join(ssd_dir, f"SelfDiff_cycle_{i}_aug.csv")
        if os.path.exists(aug_file):
            df_aug = pd.read_csv(aug_file)
            sample_sizes[str(i)] = len(df_aug)
            print(f"SSD Cycle {i}: {len(df_aug)} samples added")
    return sample_sizes


def run_control_cycle(cycle, prev_cycle, n_samples, ssd_dir, output_dir, pca_model,
                      tokenizer, model, collator, is_first_cycle=False, random_state=42):
    """
    Run a control cycle with random sampling instead of threshold-based filtering.
    
    Args:
        cycle: Current cycle number (string)
        prev_cycle: Previous cycle identifier
        n_samples: Number of samples to randomly select (matched to SSD)
        ssd_dir: Directory with SSD results (for PCA model)
        output_dir: Output directory for control results
        pca_model: PCA model from SSD experiment
        is_first_cycle: Whether this is the first student cycle
        random_state: Random seed for reproducibility
    """
    print("\n" + "="*60)
    print(f"CONTROL CYCLE: {cycle} (Random Sampling, n={n_samples})")
    print("="*60)
    
    # Load previous cycle data
    if prev_cycle == "exp":
        df_prev = pd.read_csv(os.path.join(ssd_dir, f"SelfDiff_cycle_{prev_cycle}.csv"))
    else:
        df_prev = pd.read_csv(os.path.join(output_dir, f"SelfDiff_cycle_{prev_cycle}.csv"))
    
    if is_first_cycle:
        # First cycle: load MD simulation data
        df_mol = pd.read_csv("../data/processed/SelfDiff_MDmolecules.csv")
        id2smiles = df_mol.set_index("MD_ID")["can_SMILES"].to_dict()
        
        df_add = pd.read_csv("../data/MD/selfdiff/gaff2/diffusion_coefficients_adaptive_full.csv")
        df_add["can_SMILES"] = df_add["molecule"].map(id2smiles)
        df_add["T_K"] = 300
        df_add = df_add.rename(columns={"molecule": "MD_ID", "D_m2_s": "Dmd_m2_s"})
        df_add["Dmd×10-9/m2·s-1"] = df_add["Dmd_m2_s"] * 1e9
        df_add = df_add[~df_add["can_SMILES"].isin(df_prev["can_SMILES"])]
    else:
        df_add = pd.read_csv(os.path.join(output_dir, f"SelfDiff_cycle_{prev_cycle}_leftover.csv"))
        df_add = df_add[~df_add["can_SMILES"].isin(df_prev["can_SMILES"])]
    
    print(f"Previous data: {len(df_prev)}, Available candidates: {len(df_add)}")
    
    if len(df_add) == 0 or n_samples == 0:
        print("No samples to add. Stopping.")
        return None
    
    # Embed SMILES for prediction (needed to get Dpred for consistency)
    embedded_smiles_add = batch_embed_smiles(
        df_add["can_SMILES"].tolist(), output_dir, cycle,
        tokenizer, model, collator, prefix="df_add"
    )
    
    # Prepare features and predict (for logging purposes)
    combined_add, _ = prepare_features(df_add, embedded_smiles_add, pca_model)
    
    if prev_cycle == "exp":
        prev_model = joblib.load(os.path.join(ssd_dir, f"SelfDiff_RT_rf_roberta_best_cycle_{prev_cycle}.joblib"))
    else:
        prev_model = joblib.load(os.path.join(output_dir, f"SelfDiff_RT_rf_roberta_best_cycle_{prev_cycle}.joblib"))
    
    y_pred = prev_model.predict(combined_add)
    df_add["Dpred×10-9/m2·s-1"] = y_pred
    
    # RANDOM SAMPLING instead of threshold-based filtering
    n_to_sample = min(n_samples, len(df_add))
    df_aug = df_add.sample(n=n_to_sample, random_state=random_state)
    df_leftover = df_add.drop(df_aug.index)
    
    df_aug["ln(Dmd×10-9/m2·s-1)"] = np.log(df_aug["Dmd×10-9/m2·s-1"])
    df_aug["ln(Dpred×10-9/m2·s-1)"] = np.log(df_aug["Dpred×10-9/m2·s-1"])
    df_aug["ref"] = "MD"
    
    print(f"Randomly sampled: {len(df_aug)}")
    print(f"Leftover: {len(df_leftover)}")
    
    df_aug.to_csv(os.path.join(output_dir, f"SelfDiff_cycle_{cycle}_aug.csv"), index=False)
    df_leftover.to_csv(os.path.join(output_dir, f"SelfDiff_cycle_{cycle}_leftover.csv"), index=False)
    
    # Combine with previous data
    df = pd.concat([df_prev, df_aug])
    cols = ["can_SMILES", "T_K", "Dexp×10-9/m2·s-1", "box", "window_ps",
            "Dmd×10-9/m2·s-1", "Dpred×10-9/m2·s-1", "ref"]
    df = df[[c for c in cols if c in df.columns]]
    
    df["target"] = df["Dexp×10-9/m2·s-1"].fillna(df["Dpred×10-9/m2·s-1"])
    
    both_present = df["Dexp×10-9/m2·s-1"].notna() & df["Dpred×10-9/m2·s-1"].notna()
    assert not both_present.any(), "Found rows with both Dexp and Dpred!"
    
    df.to_csv(os.path.join(output_dir, f"SelfDiff_cycle_{cycle}.csv"), index=False)
    
    # Embed all SMILES
    embedded_smiles = batch_embed_smiles(
        df["can_SMILES"].tolist(), output_dir, cycle,
        tokenizer, model, collator, prefix="df"
    )
    
    # Prepare features
    combined_embeddings, embeddings_reduced = prepare_features(df, embedded_smiles, pca_model)
    
    with open(os.path.join(output_dir, f"embeddings_roberta_reduced_cycle_{cycle}.pkl"), "wb") as f:
        pickle.dump({"embeddings": embeddings_reduced, "cycle": cycle}, f)
    
    # Train model
    y = df["target"]
    best_model, metrics, grid_search, cv_results = train_rf_model(
        combined_embeddings, y, output_dir, cycle
    )
    
    save_cycle_results(df, y, metrics, grid_search, cv_results, output_dir, cycle)
    joblib.dump(best_model, os.path.join(output_dir, f"SelfDiff_RT_rf_roberta_best_cycle_{cycle}.joblib"))
    
    return len(df_leftover)


# Main

def main():
    parser = argparse.ArgumentParser(description="Run control experiment for SSD pipeline")
    parser.add_argument("--ssd_dir", type=str, required=True,
                        help="Directory with completed SSD results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: {ssd_dir}_control)")
    parser.add_argument("--cycles", type=int, default=4,
                        help="Number of cycles to run (default: 4)")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = f"{args.ssd_dir}_control"
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"SSD directory: {args.ssd_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Cycles: {args.cycles}")
    print(f"Random state: {args.random_state}")
    
    # Get sample sizes from SSD run
    print("\nGetting sample sizes from SSD run...")
    sample_sizes = get_ssd_sample_sizes(args.ssd_dir, args.cycles)
    
    if not sample_sizes:
        print("ERROR: No SSD augmentation files found. Run SSD first.")
        return
    
    # Load PCA model from SSD
    pca_path = os.path.join(args.ssd_dir, "pca_object_cycle_exp.pkl")
    with open(pca_path, "rb") as f:
        pca_model = pickle.load(f)
    print(f"Loaded PCA model from {pca_path}")
    
    # Load RoBERTa model
    print("\nLoading RoBERTa model...")
    tokenizer, model, collator = load_roberta_model()
    
    # Run control cycles
    prev_cycle = "exp"
    for i in range(1, args.cycles + 1):
        cycle = str(i)
        
        if cycle not in sample_sizes:
            print(f"\nNo sample size found for cycle {cycle}. Stopping.")
            break
        
        leftover = run_control_cycle(
            cycle, prev_cycle, sample_sizes[cycle],
            args.ssd_dir, args.output_dir, pca_model,
            tokenizer, model, collator,
            is_first_cycle=(i == 1),
            random_state=args.random_state
        )
        
        if leftover is None or leftover == 0:
            print(f"\nStopping early: no more molecules after cycle {cycle}")
            break
        
        prev_cycle = cycle
    
    print("\n" + "="*60)
    print("Control Experiment Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
