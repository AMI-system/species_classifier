#!/bin/bash
#SBATCH --qos=turing
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --output=./regional_scripts/anguilla_train.out
#SBATCH --time=10:00:00          # total run time limit (DD-HH:MM:SS)

# 0. Set up
module purge; module load baskerville
module load bask-apps/live
module load CUDA/11.7.0
module load Python/3.9.5-GCCcore-10.3.0
module load Miniforge3/24.1.2-0
eval "$(${EBROOTMINIFORGE3}/bin/conda shell.bash hook)"
source "${EBROOTMINIFORGE3}/etc/profile.d/mamba.sh"

# Define the path to your existing Conda environment (modify as appropriate)
CONDA_ENV_PATH="/bask/projects/v/vjgo8416-amber/kg_conda_env2/"

mamba activate "${CONDA_ENV_PATH}"

# (Optional) check the CPU/GPU set up
#/bask/homes/f/fspo1218/NVIDIA_CUDA-11.1_Samples/bin/x86_64/linux/release/cudaOpenMP

# 1. create_dataset_split
# echo 'Create dataset split'
# python 01_create_dataset_split.py \
#     --data_dir /bask/homes/f/fspo1218/amber/data/gbif_download_standalone/gbif_images/ \
#     --write_dir /bask/homes/f/fspo1218/amber/data/gbif_anguilla/ \
#     --species_list /bask/homes/f/fspo1218/amber/projects/gbif_download_standalone/species_checklists/anguilla-moths-keys-nodup.csv \
#     --train_ratio 0.75 \
#     --val_ratio 0.10 \
#     --test_ratio 0.15 \
#     --filename 02_anguilla_data



# # # 2. calculate_taxa_statistics

# # mkdir -p /bask/homes/f/fspo1218/amber/data/gbif_anguilla/

# python 02_calculate_taxa_statistics.py \
#     --species_list /bask/homes/f/fspo1218/amber/projects/gbif_download_standalone/species_checklists/anguilla-moths-keys-nodup.csv \
#     --write_dir /bask/homes/f/fspo1218/amber/data/gbif_anguilla/ \
#     --numeric_labels_filename 02_anguilla_data_numeric_labels \
#     --taxon_hierarchy_filename 02_anguilla_data_taxon_hierarchy \
#     --category_map_filename 02_anguilla_data_category_map \
#     --training_points_filename 02_anguilla_data_count_training_points \
#     --train_split_file /bask/homes/f/fspo1218/amber/data/gbif_anguilla/02_anguilla_data-train-split.csv

# printf '\nmake sure you update ./configs/02_anguilla_data_config.json with these values!\n\n'

# # 3. create webdataset
# for VARIABLE in 'train' 'val' 'test'
# do
#     echo '--' $VARIABLE
#     mkdir -p /bask/homes/f/fspo1218/amber/data/gbif_anguilla/$VARIABLE
#     python 03_create_webdataset.py \
#         --dataset_dir /bask/homes/f/fspo1218/amber/data/gbif_download_standalone/gbif_images/ \
#         --dataset_filepath /bask/homes/f/fspo1218/amber/data/gbif_anguilla/02_anguilla_data-$VARIABLE-split.csv \
#         --label_filepath /bask/homes/f/fspo1218/amber/data/gbif_anguilla/02_anguilla_data_numeric_labels.json \
#         --image_resize 500 \
#         --max_shard_size 100000000 \
#         --webdataset_pattern "/bask/homes/f/fspo1218/amber/data/gbif_anguilla/$VARIABLE/$VARIABLE-500-%06d.tar"
# done


# 4. Train the model
generate_file_range() {
    local directory="/bask/homes/f/fspo1218/amber/data/gbif_anguilla"
    local prefix="$1"  

    # Count the number of files matching the specified prefix in the directory
    local file_count=$(ls -1 "$directory"/"$prefix"/"$prefix"-500* 2>/dev/null | wc -l)
    ((file_count--))

    file_count=$(printf "%06d" "$file_count")
    formatted_url="$directory/$prefix/$prefix-500-{000000..$file_count}.tar"

    echo $formatted_url
}

# Example usage:
train_url=$(generate_file_range "train")
test_url=$(generate_file_range "test")
val_url=$(generate_file_range "val")

echo 'Training the model'
python 04_train_model.py  \
    --train_webdataset_url "$train_url" \
    --val_webdataset_url "$val_url" \
    --test_webdataset_url "$test_url" \
    --config_file ./configs/02_anguilla_data_config.json \
    --dataloader_num_workers 6 \
    --random_seed 42

# ding ding
echo $'\a'
