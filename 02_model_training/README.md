# Model Training

This model is built using tensorflow. The user needs to run the following scripts in a sequence to train the model:

## 1. `01-create_dataset_split.py`

This creates training, validation and testing splits of the data downloaded from GBIF.

```bash
python 01-create_dataset_split.py \
    --data_dir ../01_data_download/output_data/gbif_data/ \
    --write_dir ./data/ \
    --train_ratio 0.75 \
    --val_ratio 0.10 \
    --test_ratio 0.15 \
    --filename 01_uk_moth_data
```

The description of the arguments to the script:
* `--data_dir`: Path to the root directory containing the GBIF data. **Required**.
* `--write_dir`: Path to the directory for saving the split files. **Required**.
* `--train_ratio`: Proportion of data for training. **Required**.
* `--val_ratio`: Proportion of data for validation. **Required**.
* `--test_ratio`: Proportion of data for testing. **Required**.
* `--filename`: Initial name for the split files. **Required**.

<br>
