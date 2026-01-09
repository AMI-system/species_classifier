#!/bin/bash
#SBATCH --account orchid
#SBATCH --qos orchid
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --nodes 1
#SBATCH --output=classifier_japan.out
#SBATCH --partition=orchid
#SBATCH --job-name=classifier_japan
#SBATCH --mem=32G

# Module loading
source ~/miniforge3/bin/activate
conda activate species_classifier2

# # 1. create_dataset_split
# echo 'Create dataset split'
# python 01_create_dataset_split.py \
#     --data_dir  /gws/ssde/j25b/ceh_generic/kgoldmann/gbif_images/\
#     --write_dir /gws/ssde/j25b/ceh_generic/kgoldmann/classifier_data/gbif_japan/ \
#     --species_list /home/users/katriona/gbif_download_standalone/species_checklists/final/japan-moths-keys-nodup.csv \
#     --train_ratio 0.75 \
#     --val_ratio 0.10 \
#     --test_ratio 0.15 \
#     --filename 01_japan_data

# # 2. calculate_taxa_statistics
# python 02_calculate_taxa_statistics.py \
#     --species_list /home/users/katriona/gbif_download_standalone/species_checklists/final/japan-moths-keys-nodup.csv \
#     --write_dir /gws/ssde/j25b/ceh_generic/kgoldmann/classifier_data/gbif_japan/ \
#     --numeric_labels_filename 01_japan_data_numeric_labels \
#     --taxon_hierarchy_filename 01_japan_data_taxon_hierarchy \
#     --category_map_filename 01_japan_data_category_map \
#     --training_points_filename 01_japan_data_count_training_points \
#     --train_split_file /gws/ssde/j25b/ceh_generic/kgoldmann/classifier_data/gbif_japan/01_japan_data-train-split.csv

# printf '\nmake sure you update ./configs/01_japan_data_config.json with these values!\n\n'

# # 3. create webdataset
# for VARIABLE in 'train' 'val' 'test'
# do
#     echo '--' $VARIABLE
#     mkdir -p /gws/ssde/j25b/ceh_generic/kgoldmann/classifier_data/gbif_japan/$VARIABLE
#     python 03_create_webdataset.py \
#         --dataset_dir /gws/ssde/j25b/ceh_generic/kgoldmann/gbif_images/ \
#         --dataset_filepath /gws/ssde/j25b/ceh_generic/kgoldmann/classifier_data/gbif_japan/01_japan_data-$VARIABLE-split.csv \
#         --label_filepath /gws/ssde/j25b/ceh_generic/kgoldmann/classifier_data/gbif_japan/01_japan_data_numeric_labels.json \
#         --image_resize 500 \
#         --max_shard_size 100000000 \
#         --webdataset_pattern "/gws/ssde/j25b/ceh_generic/kgoldmann/classifier_data/gbif_japan/$VARIABLE/$VARIABLE-500-%06d.tar"
# done



# 4. Train the model
generate_file_range() {
    local directory="/gws/ssde/j25b/ceh_generic/kgoldmann/classifier_data/gbif_japan/"
    local prefix="$1"

    # Count the number of files matching the specified prefix in the directory
    local file_count=$(ls -1 "$directory"/"$prefix"/"$prefix"-500* 2>/dev/null | wc -l)
    ((file_count--))

    file_count=$(printf "%06d" "$file_count")
    formatted_url="$directory/$prefix/$prefix-500-{000000..$file_count}.tar"

    echo $formatted_url
}

train_url=$(generate_file_range "train")
test_url=$(generate_file_range "test")
val_url=$(generate_file_range "val")

echo 'Training the model'
python 04_train_model.py  \
    --train_webdataset_url "$train_url" \
    --val_webdataset_url "$val_url" \
    --test_webdataset_url "$test_url" \
    --config_file ./configs/01_japan_data_config.json \
    --dataloader_num_workers 6 \
    --random_seed 42

ding ding
echo $'\a'
