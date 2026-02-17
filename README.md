# Learning Diffusion from Sparse Data: A Machine-Learning Bridge between Molecular Motion and Macroscopic Transport

This repository contains the data processing pipeline, trained models, and reproducible scripts for predicting molecular self-diffusion coefficients using self-supervised distillation (SSD) with molecular dynamics simulations.

Manuscript: TBD with DOI

## Deliverables

The main output files are:
- `data/processed/SelfDiff_MDmolecules.csv` - Molecule dataset with SMILES, properties, and MD simulation IDs
- `data/MD/selfdiff/gaff2/diffusion_coefficients_adaptive_full.csv` - Diffusion coefficients extracted from MD simulations
- `data/SelfDiff_SSD_20251117_threshold1/` - Self-supervised distillation (SSD) model outputs including:
  - Trained Random Forest models (`.joblib` files) for each cycle
  - Cross-validation results and test performance metrics
  - Augmented training data from each SSD cycle
- `data/SelfDiff_SSD_20251117_threshold1_control/` - Control experiment outputs (without MD augmentation)

## Data Processing Pipeline

The notebooks in `sandbox/` should be run in the following order:

1. **`1_processing_rawDBs.ipynb`** - Processes raw databases to standardize SMILES and extract properties
2. **`2_SelfDiff_MDmolecules_from_DBs.ipynb`** - Creates the molecule dataset from processed databases
3. **`3_SelfDiff_MDsystemprep.ipynb`** - Prepares molecular systems for MD simulations (generates input files for AMBER)
4. **`diffusion.ipynb`** - Extracts diffusion coefficients from MD trajectory files
5. **`4_SelfDIff_SSD_scripted.ipynb`** - Runs the self-supervised distillation pipeline to train diffusion prediction models
6. **`4.5_SelfDiff_SSD_control.ipynb`** - Control experiment with random sampling (for comparison)

### Reproducible Python Scripts

For reproducibility, standalone Python scripts are provided in the `scripts/` directory:

```bash
# Run the main SSD pipeline
cd scripts/
python run_ssd.py --threshold 1.0 --cycles 4 --output_dir ../data/SelfDiff_SSD_output

# Run the control experiment (requires completed SSD run)
python run_ssd_control.py --ssd_dir ../data/SelfDiff_SSD_output --cycles 4
```

**Script options:**

- `run_ssd.py`:
  - `--threshold`: Agreement threshold for filtering MD predictions (default: 1.0)
  - `--cycles`: Number of student cycles to run (default: 4)
  - `--output_dir`: Output directory for results

- `run_ssd_control.py`:
  - `--ssd_dir`: Directory with completed SSD results (required)
  - `--output_dir`: Output directory (default: `{ssd_dir}_control`)
  - `--cycles`: Number of cycles to run (default: 4)
  - `--random_state`: Random seed for reproducibility (default: 42)

## External Data Requirements

### ChEMBL Database File

The `1_processing_rawDBs.ipynb` notebook requires the ChEMBL chemical representations file which is not included due to its large size (~632 MB).

To obtain this file:
1. Download `chembl_34_chemreps.txt` from the ChEMBL database: https://www.ebi.ac.uk/chembl/
2. Place it in `data/raw/chembl_34_chemreps.txt`

### MD Trajectory Files

The `diffusion.ipynb` notebook requires MD simulation trajectory files (`.nc` and `.prmtop` files) which are not included in this repository due to their large size. These files should be placed in `data/MD/selfdiff/gaff2/simulation_Expanse/`.

### RoBERTa Embeddings

The `4_SelfDIff_SSD_scripted.ipynb` notebook uses the `entropy/roberta_zinc_480m` model from HuggingFace for molecular embeddings. The embedding parquet files are not included but will be generated when running the notebook.

## Directory Structure

```
GitHub_DermaDiff_publish/
├── README.md
├── sandbox/
│   ├── 1_processing_rawDBs.ipynb
│   ├── 2_SelfDiff_MDmolecules_from_DBs.ipynb
│   ├── 3_SelfDiff_MDsystemprep.ipynb
│   ├── 4_SelfDIff_SSD_scripted.ipynb
│   ├── 4.5_SelfDiff_SSD_control.ipynb
│   └── diffusion.ipynb
├── scripts/
│   ├── run_ssd.py              # Reproducible SSD pipeline script
│   └── run_ssd_control.py      # Reproducible control experiment script
├── data/
│   ├── raw/
│   │   ├── SelfDiff_exp_iecr_2c03342_si_001.xlsx
│   │   └── SkinPerm_exp_250707.xlsx
│   ├── processed/
│   │   ├── SelfDiff_MDmolecules.csv
│   │   ├── SelfDiff_exp_processed.csv
│   │   ├── SkinPerm_exp_250707_processed.csv
│   │   └── chembl_batch1_processed.csv
│   ├── MD/
│   │   └── selfdiff/
│   │       └── gaff2/
│   │           └── diffusion_coefficients_adaptive_full.csv
│   ├── SelfDiff_SSD_20251117_threshold1/ (SSD model outputs)
│   │   ├── SelfDiff_RT_rf_roberta_best_cycle_*.joblib (trained models)
│   │   ├── SelfDiff_cycle_*_results.csv (prediction results)
│   │   ├── SelfDiff_cycle_*_test_performance.csv (test metrics)
│   │   └── ... (other cycle outputs)
│   └── SelfDiff_SSD_20251117_threshold1_control/ (control experiment outputs)
```

## Dependencies

The notebooks and scripts require the following Python packages:
- pandas
- numpy
- rdkit
- MDAnalysis
- matplotlib
- tqdm
- pubchempy
- thermo
- scikit-learn
- transformers (for RoBERTa embeddings)
- joblib
- kneed (for PCA knee detection)
- pyarrow (for parquet file handling)
