# python3 train.py --dataroot ./datasets/mr2ct_pix2pix_nc1 --name mr2ct_pix2pix_nc1 --model pix2pix --display_id -1 --load_size 266 --input_nc 1 --output_nc 1 --n_epochs 2500 --gpu_ids -1 --dataset_mode aligned






python3 train.py --dataroot ./datasets/mr2ct_small --name mr2ct --model autoGAN --display_id -1 --load_size 256 --input_nc 1 --output_nc 1 --n_epochs 2500 --norm batch --batch_size 10 --lr 0.00005 --D_update_freq 1 --netG unet_256 --gpu_ids -1

