#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=10000M
#SBATCH -p gpu --gres=gpu:titanrtx:4
#SBATCH --time=3-24:00:00

echo "Prepping cluster, using $CUDA_VISIBLE_DEVICES as gpus"

# Navigate to folder
cd ~/autoGAN

#Activate env and ensure requirements are met
module load pytorch
module load cuda
source gan_env/bin/activate

#Run training session
python train.py --dataroot ./datasets/mr2ct --name test --model autoGAN --display_id -1 --load_size 256 --input_nc 1 --output_nc 1 --n_epochs 2500 --norm batch --batch_size 10 --lr 0.00005 --netG unet_256 --gpu_ids 0,1,2,3




