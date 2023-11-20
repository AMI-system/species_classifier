#!/bin/bash 
#SBATCH --qos=turing
#SBATCH --cpus-per-task=6      
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --output=train.out
#SBATCH --time=10:00:00          # total run time limit (DD-HH:MM:SS)

# 0. Set up

#Load the required modules
module purge; module load baskerville
module load bask-apps/live
module load CUDA/11.7.0
#module load Python/3.8.6-GCCcore-10.2.0
module load Python/3.9.5-GCCcore-10.3.0
module load Miniconda3/4.10.3
eval "$(${EBROOTMINICONDA3}/bin/conda shell.bash hook)"

conda activate ~/amber/kg_conda_env2

# (Optional) check the CPU/GPU set up 
#/bask/homes/f/fspo1218/NVIDIA_CUDA-11.1_Samples/bin/x86_64/linux/release/cudaOpenMP

# 1. create_dataset_split
echo 'Create dataset split'
python 01_create_dataset_split.py \
    --data_dir /bask/homes/f/fspo1218/amber/data/gbif_macro_data/gbif_macro/ \
    --write_dir /bask/homes/f/fspo1218/amber/data/gbif_macro_data/ \
    --species_list /bask/homes/f/fspo1218/amber/projects/gbif_download_standalone/species_checklists/singapore-moths-keys-nodup.csv \
    --train_ratio 0.75 \
    --val_ratio 0.10 \
    --test_ratio 0.15 \
    --filename 01_singapore_macro_data



# 2. calculate_taxa_statistics
python 02_calculate_taxa_statistics.py \
    --species_list /bask/homes/f/fspo1218/amber/projects/gbif_download_standalone/species_checklists/singapore-moths-keys-nodup.csv \
    --write_dir /bask/homes/f/fspo1218/amber/data/gbif_macro_data/ \
    --numeric_labels_filename 01_uk_macro_data_numeric_labels \
    --taxon_hierarchy_filename 01_uk_macro_data_taxon_hierarchy \
    --training_points_filename 01_uk_macro_data_count_training_points \
    --train_split_file /bask/homes/f/fspo1218/amber/data/gbif_macro_data/01_singapore_macro_data-train-split.csv

# 5. Train the model
# python 04_train_model.py  \
#     --train_webdataset_url "/bask/projects/v/vjgo8416-amber/data/binary_moth_training/gbif_crop_ready/datasets/macro/train/train-500-{000000..000009}.tar" \
#     --val_webdataset_url "/bask/projects/v/vjgo8416-amber/data/binary_moth_training/gbif_crop_ready/datasets/macro/val/val-500-{000000..000001}.tar" \
#     --test_webdataset_url "/bask/projects/v/vjgo8416-amber/data/binary_moth_training/gbif_crop_ready/datasets/macro/test/test-500-{000000..000001}.tar" \
#     --config_file ./configs/01_uk_macro_data_config_cropped.json \
#     --dataloader_num_workers 6 \
#     --random_seed 42

# ding ding
echo $'\a'