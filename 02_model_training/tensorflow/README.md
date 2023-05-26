# Tensorflow Model Training

This model is built using tensorflow. The user needs to run the following scripts in a sequence to train the model:

## 1. `01-create_dataset_split.py`

This creates training, validation and testing splits of the data downloaded from GBIF.

```bash
python 01_create_dataset_split.py \
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

## 2. `02-calculate_taxa_statistics.py`

This calculates information and statistics regarding the taxonomy to be used for model training.

```bash
python 02_calculate_taxa_statistics.py \
    --species_list ../01_data_download/uksi-moths.csv \
    --write_dir ./data/ \
    --numeric_labels_filename 01_uk_moth_data_numeric_labels \
    --taxon_hierarchy_filename 01_uk_moth_data_taxon_hierarchy \
    --training_points_filename 01_uk_moth_data_count_training_points \
    --train_split_file ./data/01_uk_moth_data-train-split.csv
```

The description of the arguments to the script:
* `--species_list`: Path to the species list. **Required**.
* `--write_dir`: Path to the directory for saving the information. **Required**.
* `--numeric_labels_filename`: Filename for numeric labels file. **Required**.
* `--taxon_hierarchy_filename`: Filename for taxon hierarchy file. **Required**.
* `--training_points_filename`: Filename for storing the count of training points. **Required**.
* `--train_split_file`: Path to the training split file. **Required**.

<br>
