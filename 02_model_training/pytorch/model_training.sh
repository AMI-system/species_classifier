#!/bin/bash 
#SBATCH --qos=turing
#SBATCH --cpus-per-task=6      
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --output=train.out
#SBATCH --time=10:00:00          # total run time limit (HH:MM:SS)

# 1. Load the virtual environment

module purge
module load baskerville
#module load bask-apps/live
#module load fosscuda/2020b
module load CUDA/11.7.0
module load Python/3.8.6-GCCcore-10.2.0

source /bask/projects/v/vjgo8416-amber/venv/kg_py_3.8/bin/activate

# pip3 install torch torchvision torchaudio -U
pip install -r /bask/homes/f/fspo1218/amber/projects/on_device_classifier/requirements.txt

# 2. Create the datasets
# for VARIABLE in 'val' 'test' 'train'
# do
#     echo $VARIABLE
#     # mkdir -p ./data/datasets/macro/$VARIABLE
#     # python 03-create_webdataset.py \
#     #     --dataset_dir ../../01_data_download/output_data/gbif_macro/ \
#     #     --dataset_filepath ./data/01_uk_macro_data-$VARIABLE-split.csv \
#     #     --label_filepath ./data/01_uk_macro_data_numeric_labels.json \
#     #     --image_resize 500 \
#     #     --max_shard_size 100000000 \
#     #     --webdataset_pattern "./data/datasets/macro/$VARIABLE/$VARIABLE-500-%06d.tar"
# done

#python dummy_python.py

# 3. Train the model
CUDA_LAUNCH_BLOCKING=1 python 04_train_model.py  \
    --train_webdataset_url "../../../../data/gbif_macro_data/datasets/macro/train/train-500-{000000..000067}.tar" \
    --val_webdataset_url "../../../../data/gbif_macro_data/datasets/macro/val/val-500-{000000..000009}.tar" \
    --test_webdataset_url "../../../../data/gbif_macro_data/datasets/macro/test/test-500-{000000..000013}.tar" \
    --config_file ./configs/01_uk_macro_data_config.json \
    --dataloader_num_workers 6 \
    --random_seed 42


echo $'\a'