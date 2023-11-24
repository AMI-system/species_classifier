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


# 1. create_dataset_split
echo 'Create dataset split'
python 01_create_dataset_split.py \
    --data_dir /bask/homes/f/fspo1218/amber/data/gbif_download_standalone/gbif_images/ \
    --write_dir /bask/homes/f/fspo1218/amber/data/gbif_costarica/ \
    --species_list /bask/homes/f/fspo1218/amber/projects/gbif_download_standalone/species_checklists/costarica-moths-keys-nodup.csv \
    --train_ratio 0.75 \
    --val_ratio 0.10 \
    --test_ratio 0.15 \
    --filename 01_costarica_data

# 2. calculate_taxa_statistics
# python 02_calculate_taxa_statistics.py \
#     --species_list /bask/homes/f/fspo1218/amber/projects/gbif_download_standalone/species_checklists/costarica-moths-keys-nodup.csv \
#     --write_dir /bask/homes/f/fspo1218/amber/data/gbif_costarica/ \
#     --numeric_labels_filename 01_costarica_data_numeric_labels \
#     --taxon_hierarchy_filename 01_costarica_data_taxon_hierarchy \
#     --training_points_filename 01_costarica_data_count_training_points \
#     --train_split_file /bask/homes/f/fspo1218/amber/data/gbif_costarica/01_costarica_data-train-split.csv

# printf '\nmake sure you update ./configs/01_costarica_data_config.json with these values!\n\n'

# 3. create webdataset
# for VARIABLE in 'train' 'val' 'test'
# do
#     echo '--' $VARIABLE
#     mkdir -p /bask/homes/f/fspo1218/amber/data/gbif_costarica/$VARIABLE
#     python 03_create_webdataset.py \
#         --dataset_dir /bask/homes/f/fspo1218/amber/data/gbif_download_standalone/gbif_images/ \
#         --dataset_filepath /bask/homes/f/fspo1218/amber/data/gbif_costarica/01_costarica_data-$VARIABLE-split.csv \
#         --label_filepath /bask/homes/f/fspo1218/amber/data/gbif_costarica/01_costarica_data_numeric_labels.json \
#         --image_resize 500 \
#         --max_shard_size 100000000 \
#         --webdataset_pattern "/bask/homes/f/fspo1218/amber/data/gbif_costarica/$VARIABLE/$VARIABLE-500-%06d.tar"
# done

# make sure you update ./configs/01_costarica_data_config.json

# # 4. Train the model
# echo 'Training the model'
# python 04_train_model.py  \
#     --train_webdataset_url "/bask/homes/f/fspo1218/amber/data/gbif_costarica/train/train-500-{000000..000160}.tar" \
#     --val_webdataset_url "/bask/homes/f/fspo1218/amber/data/gbif_costarica/val/val-500-{000000..000021}.tar" \
#     --test_webdataset_url "/bask/homes/f/fspo1218/amber/data/gbif_costarica/test/test-500-{000000..000031}.tar" \
#     --config_file ./configs/01_costarica_data_config.json \
#     --dataloader_num_workers 6 \
#     --random_seed 42


# ding ding
echo $'\a'