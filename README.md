# Fractional-order-Differential-Equations-Super-Resolution
the code for FDE-Net: A Memory-efficiency Densely Connected Network Inspired from Fractional-order Differential Equations for Single Image Super-Resolution

train
python main.py --save fde_x4_alpha0.1_9_2conv_8block_64 --scale 4 --n_resblocks 8 --n_feats 64 --gpu-id 7,6,2 --n_GPUs 3 --model EDSR_FDE --patch_size 192 --epoch 800 --lr_decay 250 --alpha 0.1 --num_for 9 

test
python main.py --data_test Set5+Set14+B100+Urban100 --data_range 801-900 --scale 4 --test_only --ext bin --n_resblocks 8 --n_feats 64 --gpu-id 4 --n_GPUs 1 --res_scale 0.1 --pre_train /data0/experiment/fde_x4_alpha0.1_9_2conv_8block_64/model/model_best.pt --save_results --model edsr_fde  --alpha 0.1 --num_for 9
