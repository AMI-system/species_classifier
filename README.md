# Species Classifier Models

This repo creates PyTorch species classification models based on GBIF images (see the [gbif_download_standalone](https://github.com/AMI-system/gbif_download_standalone) repo for information and code to downloading images).  

This model is built using pytorch. The user needs to run the following scripts in a sequence to train the model:


## Training the Models for a Given Region

The easiest way to run this pipeline is to use the `regional_scripst/{region}_model.sh` files. 


To run this for a given species list with Slurm (e.g., on [Baskerville](https://docs.baskerville.ac.uk/)). For example `sbatch regional_scripst/costarica_model.sh`, which will output to `./regional_scripst/cr_train.out`.

### Scripts

The pipeline is comprised of 4 scripts: 

#### **`01-create_dataset_split.py`**

This creates training, validation and testing splits of the data downloaded from GBIF.

```bash
python 01_create_dataset_split.py \
    --data_dir /bask/homes/f/fspo1218/amber/data/gbif_download_standalone/gbif_images/ \
    --write_dir /bask/homes/f/fspo1218/amber/data/gbif_costarica/ \
    --species_list /bask/homes/f/fspo1218/amber/projects/gbif_download_standalone/species_checklists/costarica-moths-keys-nodup.csv \
    --train_ratio 0.75 \
    --val_ratio 0.10 \
    --test_ratio 0.15 \
    --filename 01_costarica_data
```

   The description of the arguments to the script:
* `--data_dir`: Path to the root directory containing the GBIF data. **Required**.
* `--write_dir`: Path to the directory for saving the split files. **Required**.
* `--train_ratio`: Proportion of data for training. **Required**.
* `--val_ratio`: Proportion of data for validation. **Required**.
* `--test_ratio`: Proportion of data for testing. **Required**.
* `--filename`: Initial name for the split files. **Required**.
* `--species_list`: Path to the species list. **Required**.



#### **`02_calculate_taxa_statistics.py`**

This calculates information and statistics regarding the taxonomy to be used for model training.

```bash
python 02_calculate_taxa_statistics.py \
    --species_list /bask/homes/f/fspo1218/amber/projects/gbif_download_standalone/species_checklists/costarica-moths-keys-nodup.csv \
    --write_dir /bask/homes/f/fspo1218/amber/data/gbif_costarica/ \
    --numeric_labels_filename 01_costarica_data_numeric_labels \
    --taxon_hierarchy_filename 01_costarica_data_taxon_hierarchy \
    --training_points_filename 01_costarica_data_count_training_points \
    --train_split_file /bask/homes/f/fspo1218/amber/data/gbif_costarica/01_costarica_data-train-split.csv
```

The description of the arguments to the script:
  - `--species_list`: Path to the species list. **Required**.
  - `--write_dir`: Path to the directory for saving the information. **Required**.
- `--numeric_labels_filename`: Filename for numeric labels file. **Required**.
* `--taxon_hierarchy_filename`: Filename for taxon hierarchy file. **Required**.
* `--training_points_filename`: Filename for storing the count of training points. **Required**.
* `--train_split_file`: Path to the training split file. **Required**.

**THEN** after this is done you need to add the number fo families, genus, and species to the `./configs/01_uk_macro_data_config.json` file. This is done manually.

#### **`03_create_webdataset.py`**

Creates webdataset from raw image data. It needs to be run individually for each of the train, validation and test sets.

So we will loop through each set:

```bash
for VARIABLE in 'train' 'val' 'test'
do
    echo '--' $VARIABLE
    mkdir -p /bask/homes/f/fspo1218/amber/data/gbif_costarica/$VARIABLE
    python 03_create_webdataset.py \
        --dataset_dir /bask/homes/f/fspo1218/amber/data/gbif_download_standalone/gbif_images/ \
        --dataset_filepath /bask/homes/f/fspo1218/amber/data/gbif_costarica/01_costarica_data-$VARIABLE-split.csv \
        --label_filepath /bask/homes/f/fspo1218/amber/data/gbif_costarica/01_costarica_data_numeric_labels.json \
        --image_resize 500 \
        --max_shard_size 100000000 \
        --webdataset_pattern "/bask/homes/f/fspo1218/amber/data/gbif_costarica/$VARIABLE/$VARIABLE-500-%06d.tar"
done
```

#### **`04_train_model.py`**

Training the Pytorch model. This step required the use of [wandb](https://wandb.ai/site). The user needs to create an account and login to the platform. The user will then need to set up a project and pass the `entity` (username) and `project` into the config file. This can be run with nohup:

```bash
nohup sh -c 'python 04_train_model.py  \
    --train_webdataset_url "$train_url" \
    --val_webdataset_url "$val_url" \
    --test_webdataset_url "$test_url" \
    --config_file ./configs/01_costarica_data_config.json \
    --dataloader_num_workers 6 \
    --random_seed 42' &
```


The description of the arguments to the script:

* `--train_webdataset_url`: path to webdataset tar files for training
* `--val_webdataset_url`: path to webdataset tar files for validation
* `--test_webdataset_url`: path to webdataset tar files for testing
* `--config_file`: path to configuration file containing training information
* `--dataloader_num_workers`: number of cpus available
* `--random_seed`: random seed for reproducible experiments

*For setting up the config file*: The total families, genuses, and species are spit out at the end of `02_calculate_taxa_statistics.py` so you can use this info to fill in the config lines 5-7.


