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

pip install huggingface_hub

# Load the virtual environment
#source /bask/projects/v/vjgo8416-amber/venv/kg_py_3.8/bin/activate

# (Optional) check the CPU/GPU set up 
#/bask/homes/f/fspo1218/NVIDIA_CUDA-11.1_Samples/bin/x86_64/linux/release/cudaOpenMP

# (Optional) load the python packages, if required
#pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
#pip3 install torch torchvision torchaudio -U
#pip install -r /bask/homes/f/fspo1218/amber/projects/on_device_classifier/requirements.txt
#pip install timm


# 5. Train the model
python 04_train_model.py  \
    --train_webdataset_url "/bask/projects/v/vjgo8416-amber/data/binary_moth_training/gbif_crop_ready/datasets/macro/train/train-500-{000000..000009}.tar" \
    --val_webdataset_url "/bask/projects/v/vjgo8416-amber/data/binary_moth_training/gbif_crop_ready/datasets/macro/val/val-500-{000000..000001}.tar" \
    --test_webdataset_url "/bask/projects/v/vjgo8416-amber/data/binary_moth_training/gbif_crop_ready/datasets/macro/test/test-500-{000000..000001}.tar" \
    --config_file ./configs/01_uk_macro_data_config_cropped.json \
    --dataloader_num_workers 6 \
    --random_seed 42

# ding ding
echo $'\a'