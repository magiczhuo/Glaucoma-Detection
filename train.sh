export CUDA_VISIBLE_DEVICES=0,1,2,3;
python train.py --dataroot /root/ZYZ/GRINLAB/dataset --data_aug --aug_prob 0.5 --batch_size 16 --num_threads 16 --gpu_ids 0,1,2,3 --name 1-resnet152-3b-3cls --model_name 3branch  --save_epoch_freq 5 --lr 0.0001 --mode 3cls

# python train.py --dataroot /root/ZYZ/GRINLAB/dataset --continue_train --epoch latest --epoch_count 3 --data_aug --aug_prob 0.5 --batch_size 16 --num_threads 16 --gpu_ids 0,1,2,3 --name 1-resnet152-3b-3cls-f12 --model_name 3branch-rcbam  --save_epoch_freq 5 --lr 0.00005 --mode 3cls