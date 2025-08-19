# PreprocessingForAdaGCL/

Scripts to convert KGCL-format datasets to AdaGCL-compatible pickle files.

## Usage

Run `python convert_adagcl.py` to produce `trnMat.pkl` / `tstMat.pkl` files exactly as used by AdaGCL.

## What it does

- Builds sparse matrices with legacy axis convention (users=rows, items=cols)
- Uses same ID mapping policy as legacy (integers from raw tokens)
- Uses same dtype for data (np.float64 as in legacy)
- Outputs exactly: `trnMat.pkl` and `tstMat.pkl`
- Creates scipy.sparse.coo_matrix objects

## Output locations

- **Primary**: `AdaGCL/Datasets/<dataset>/` (if exists)
- **Fallback**: `data/processed/adagcl/<dataset>/`

## Supported datasets

- yelp2018
- amazon-book
- mindreader_fold_0
- mindreader_fold_1
- mindreader_fold_2
- mindreader_fold_3
- mindreader_fold_4

## Configuration

Edit the config section at the top of `convert_adagcl.py` to:

- Add/remove datasets from `DATASETS` list
- Change `DATA_ROOT` path (currently `../data`)
- Modify `OUT_ROOT` path

## Matrix format

- **Shape**: (n_users, n_items)
- **Data type**: np.float64
- **Sparse format**: scipy.sparse.coo_matrix
- **Values**: 1.0 for all interactions (binary matrix)
