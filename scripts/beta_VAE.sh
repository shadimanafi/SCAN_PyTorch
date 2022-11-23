#! /bin/sh

python main.py --dataset celeba\
    --SCAN --phase beta_VAE\
    --seed 7 --lr 1e-4 --batch_size 100 --max_iter 2e6 --beta 53\
    --DAE_env_name DAE --DAE_z_dim 100\
    --beta_VAE_env_name beta_VAE --beta_VAE_z_dim 32
