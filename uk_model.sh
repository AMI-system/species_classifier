#!/bin/bash 
#SBATCH --qos=turing
#SBATCH --cpus-per-task=6      
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --output=train.out
#SBATCH --time=30:00:00          # total run time limit (DD-HH:MM:SS)

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

#pip install huggingface_hub

# Load the virtual environment
#source /bask/projects/v/vjgo8416-amber/venv/kg_py_3.8/bin/activate

# (Optional) check the CPU/GPU set up 
#/bask/homes/f/fspo1218/NVIDIA_CUDA-11.1_Samples/bin/x86_64/linux/release/cudaOpenMP

# (Optional) load the python packages, if required
#pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
#pip3 install torch torchvision torchaudio -U
#pip install -r /bask/homes/f/fspo1218/amber/projects/on_device_classifier/requirements.txt
#pip install timm



# 1. create_dataset_split
# echo 'Create dataset split'
# python 01_create_dataset_split.py \
#     --data_dir /bask/homes/f/fspo1218/amber/data/gbif_download_standalone/gbif_images/ \
#     --write_dir /bask/homes/f/fspo1218/amber/data/gbif_uk/ \
#     --species_list /bask/homes/f/fspo1218/amber/projects/gbif_download_standalone/species_checklists/uksi-moths-keys-nodup.csv \
#     --train_ratio 0.75 \
#     --val_ratio 0.10 \
#     --test_ratio 0.15 \
#     --filename 03_uk_data

# 2. calculate_taxa_statistics
# python 02_calculate_taxa_statistics.py \
#     --species_list /bask/homes/f/fspo1218/amber/projects/gbif_download_standalone/species_checklists/uksi-moths-keys-nodup.csv \
#     --write_dir /bask/homes/f/fspo1218/amber/data/gbif_uk/ \
#     --numeric_labels_filename 03_uk_data_numeric_labels \
#     --taxon_hierarchy_filename 03_uk_data_taxon_hierarchy \
#     --category_map_filename 03_uk_data_category_map \
#     --training_points_filename 03_uk_data_count_training_points \
#     --train_split_file /bask/homes/f/fspo1218/amber/data/gbif_uk/03_uk_data-train-split.csv

# printf '\nmake sure you update ./configs/03_uk_data_config.json with these values!\n\n'

# # 3. create webdataset
# for VARIABLE in 'test' 'val' #'train'
# do
#     echo '--' $VARIABLE
#     mkdir -p /bask/homes/f/fspo1218/amber/data/gbif_uk/$VARIABLE
#     python 03_create_webdataset.py \
#         --dataset_dir /bask/homes/f/fspo1218/amber/data/gbif_download_standalone/gbif_images/ \
#         --dataset_filepath /bask/homes/f/fspo1218/amber/data/gbif_uk/03_uk_data-$VARIABLE-split.csv \
#         --label_filepath /bask/homes/f/fspo1218/amber/data/gbif_uk/03_uk_data_numeric_labels.json \
#         --image_resize 500 \
#         --max_shard_size 100000000 \
#         --webdataset_pattern "/bask/homes/f/fspo1218/amber/data/gbif_uk/$VARIABLE/$VARIABLE-500-%06d.tar"
# done




# 4. Train the model
generate_file_range() {
    local directory="/bask/homes/f/fspo1218/amber/data/gbif_uk"
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
    --config_file ./configs/03_uk_data_config.json \
    --dataloader_num_workers 6 \
    --random_seed 42


# ding ding
echo $'\a'