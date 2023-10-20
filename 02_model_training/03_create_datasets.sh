for VARIABLE in 'train' 'val' 'test'
do
    echo $VARIABLE
    mkdir -p /bask/homes/f/fspo1218/amber/data/binary_moth_training/gbif_crop_ready/datasets/macro/$VARIABLE
    python 03_create_webdataset.py \
        --dataset_dir /bask/homes/f/fspo1218/amber/data/binary_moth_training/gbif_crops/ \
        --dataset_filepath /bask/homes/f/fspo1218/amber/data/binary_moth_training/gbif_crop_ready/01_uk_macro_data-$VARIABLE-split.csv \
        --label_filepath /bask/homes/f/fspo1218/amber/data/binary_moth_training/gbif_crop_ready/01_uk_macro_data_numeric_labels.json \
        --image_resize 500 \
        --max_shard_size 100000000 \
        --webdataset_pattern "/bask/homes/f/fspo1218/amber/data/binary_moth_training/gbif_crop_ready/datasets/macro/$VARIABLE/$VARIABLE-500-%06d.tar"
done

# for VARIABLE in 'train' 'val' 'test'
# do
#     echo $VARIABLE
#     mkdir -p ../../../../data/gbif_macro_data/datasets/macro/$VARIABLE
#     python 03_create_webdataset.py \
#         --dataset_dir ../../../../data/gbif_macro_data/gbif_macro/ \
#         --dataset_filepath ../../../../data/gbif_macro_data//01_uk_macro_data-$VARIABLE-split.csv \
#         --label_filepath ../../../../data/gbif_macro_data/01_uk_macro_data_numeric_labels.json \
#         --image_resize 500 \
#         --max_shard_size 100000000 \
#         --webdataset_pattern "../../../../data/gbif_macro_data/datasets/macro/$VARIABLE/$VARIABLE-500-%06d.tar"
# done