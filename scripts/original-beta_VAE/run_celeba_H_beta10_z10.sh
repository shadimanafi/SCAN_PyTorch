#! /bin/sh

python main.py --dataset celeba\
    --seed 7 --lr 1e-4 --batch_size 64 --max_iter 1e6 --beta 4\
    --beta_VAE_env_name original_beta_VAE --beta_VAE_z_dim 32
