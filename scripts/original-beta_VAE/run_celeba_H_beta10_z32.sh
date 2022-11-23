#! /bin/sh

python main.py --dataset celeba --seed 1 --lr 1e-4 --beta1 0.9 --beta2 0.999 \
    --objective H --model H --batch_size 64 --beta_VAE_z_dim 32 --max_iter 1.5e6 \
    --beta 10 --beta_VAE_env_name celeba_H_beta10_z32
