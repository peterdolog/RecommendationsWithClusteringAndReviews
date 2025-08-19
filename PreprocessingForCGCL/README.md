# PreprocessingForCGCL/

Scripts to convert KGCL-format datasets to RecBole format for CGCL compatibility.

## Usage

Run `python convert_recbole.py` to produce `train.inter` / `test.inter` files exactly as used by CGCL.

## What it does

- Parses KGCL-format files (user item1 item2 ...) into individual (user, item) pairs
- Preserves original token strings (no ID remapping)
- Writes TSV files with headers: `user_id:token`, `item_id:token`
- Outputs exactly: `train.inter` and `test.inter`
- Maintains row order and preserves duplicates

## Output locations

- **Primary**: `CGCL-Pytorch-master/dataset/<dataset>/` (if exists)
- **Fallback**: `data/processed/recbole/<dataset>/`

## Supported datasets

- yelp2018
- amazon-book
- mindreader_fold_0
- mindreader_fold_1
- mindreader_fold_2
- mindreader_fold_3
- mindreader_fold_4

## Configuration

Edit the config section at the top of `convert_recbole.py` to:

- Add/remove datasets from `DATASETS` list
- Change `DATA_ROOT` path (currently `../data`)
- Modify `CGCL_DATASET_ROOT` path
