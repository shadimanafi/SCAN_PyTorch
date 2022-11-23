#! /bin/sh

python main.py --dataset celeba\
    --SCAN --phase SCAN\
    --seed 7 --lr 1e-4 --batch_size 16 --max_iter 2e6 --beta 0.01 --Lambda 30 --display_save_step 10000\
    --DAE_env_name DAE --DAE_z_dim 100\
    --beta_VAE_env_name beta_VAE --beta_VAE_z_dim 32\
    --SCAN_env_name SCAN --SCAN_z_dim 32
