# Pytorch Model Training

This model is built using pytorch. The user needs to run the following scripts in a sequence to train the model:

## 1. `01-create_dataset_split.py`

This creates training, validation and testing splits of the data downloaded from GBIF.

```bash
python 01_create_dataset_split.py \
    --data_dir ../../../../data/gbif_macro_data/gbif_macro/ \
    --write_dir ../../../../data/gbif_macro_data/ \
    --train_ratio 0.75 \
    --val_ratio 0.10 \
    --test_ratio 0.15 \
    --filename 01_uk_macro_data
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
    --species_list ../../01_data_download/output_data/keys/uksi-macro_data.csv \
    --write_dir ../../../../data/gbif_macro_data/ \
    --numeric_labels_filename 01_uk_macro_data_numeric_labels \
    --taxon_hierarchy_filename 01_uk_macro_data_taxon_hierarchy \
    --training_points_filename 01_uk_macro_data_count_training_points \
    --train_split_file ../../../../data/gbif_macro_data/01_uk_macro_data-train-split.csv
```

The description of the arguments to the script:
* `--species_list`: Path to the species list. **Required**.
* `--write_dir`: Path to the directory for saving the information. **Required**.
* `--numeric_labels_filename`: Filename for numeric labels file. **Required**.
* `--taxon_hierarchy_filename`: Filename for taxon hierarchy file. **Required**.
* `--training_points_filename`: Filename for storing the count of training points. **Required**.
* `--train_split_file`: Path to the training split file. **Required**.

**THEN** after this is done you need to add the number fo families, genus, and species to the `./configs/01_uk_macro_data_config.json` file. This is done manually.

<br>

## 3. `03_create_webdataset.py`: Creates webdataset from raw image data. It needs to be run individually for each of the train, validation and test sets.

So we will loop through each set:

```bash
nohup ./03_create_datasets.sh &
```

The description of the arguments to the script:
* `--dataset_dir`: Path to the dataset directory containing the gbif data. **Required**.
* `--dataset_filepath`: Path to the csv file containing every data point information. **Required**.
* `--label_filepath`: File path containing numerical label information. **Required**.
* `--image_resize`: Resizing image factor (size x size) **Required**.
* `--webdataset_patern`: Path and format type to save the webdataset. It needs to be passed in double quotes. **Required**.
* `--max_shard_size`: The maximum shard size in bytes. Optional. **Default** is **10^8 (100 MB)**.
* `--random_seed`: Random seed for reproducible experiments. Optional. **Default** is **42**.


Or you can run them all individually:

```bash
python 03_create_webdataset.py \
    --dataset_dir ../../../../data/gbif_macro_data/gbif_macro/ \
    --dataset_filepath ../../../../data/gbif_macro_data/01_uk_macro_data-train-split.csv \
    --label_filepath ../../../../data/gbif_macro_data/01_uk_macro_data_numeric_labels.json \
    --image_resize 500 \
    --max_shard_size 100000000 \
    --webdataset_pattern "../../../../data/gbif_macro_data/datasets/macro/train/train-500-%06d.tar"

python 03-create_webdataset.py \
    --dataset_dir ../../../../data/gbif_macro_data/gbif_macro/ \
    --dataset_filepath ../../../../data/gbif_macro_data/01_uk_macro_data-test-split.csv \
    --label_filepath ../../../../data/gbif_macro_data/01_uk_macro_data_numeric_labels.json \
    --image_resize 500 \
    --max_shard_size 100000000 \
    --webdataset_pattern "../../../../data/gbif_macro_data/datasets/macro/test/test-500-%06d.tar"

python 03-create_webdataset.py \
    --dataset_dir ../../../../data/gbif_macro_data/gbif_macro/ \
    --dataset_filepath ../../../../data/gbif_macro_data/01_uk_macro_data-val-split.csv \
    --label_filepath ../../../../data/gbif_macro_data/01_uk_macro_data_numeric_labels.json \
    --image_resize 500 \
    --max_shard_size 100000000 \
    --webdataset_pattern "../../../../data/gbif_macro_data/datasets/macro/val/val-500-%06d.tar"
```

## 4. Training the Pytorch model

This step required the use of [wandb](https://wandb.ai/site). The user needs to create an account and login to the platform. The user will then need to set up a project and pass the `entity` (username) and `project` into the config file. The user can then run either: 
- through the script `04_train_model.py`:

- using slurm (which will output to `train.out`):
    ```
    sbatch model_training.sh
    ```

- or nohup
    ```bash
    nohup sh -c 'python 04_train_model.py  \
        --train_webdataset_url "../../../../data/gbif_macro_data/datasets/macro/train/train-500-{000000..000392}.tar" \
        --val_webdataset_url "../../../../data/gbif_macro_data/datasets/macro/val/val-500-{000000..000052}.tar" \
        --test_webdataset_url "../../../../data/gbif_macro_data/datasets/macro/test/test-500-{000000..000078}.tar" \
        --config_file ./configs/01_uk_macro_data_config.json \
        --dataloader_num_workers 4 \
        --random_seed 42' &
    ```


The description of the arguments to the script:

* `--train_webdataset_url`: path to webdataset tar files for training
* `--val_webdataset_url`: path to webdataset tar files for validation
* `--test_webdataset_url`: path to webdataset tar files for testing
* `--config_file`: path to configuration file containing training information
* `--dataloader_num_workers`: number of cpus available
* `--random_seed`: random seed for reproducible experiments

*For setting up the config file* The total families, genuses, and species are spit out at the end of `02_calculate_taxa_statistics.py` so you can use this info to fill in the config lines 5-7.

Or using the jupyter notebook `pytorch_model.ipynb`. This is mostly for debugging purposes
