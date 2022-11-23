#! /bin/sh

python main.py --dataset celeba\
    --SCAN --phase DAE\
    --seed 3 --lr 1e-3 --batch_size 100 --max_iter 2e5\
    --DAE_env_name DAE --DAE_z_dim 100
