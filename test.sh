export CUDA_VISIBLE_DEVICES=0;
# python test_ukb_eval.py --model_name 3branch-rcbam --model_path 'checkpoints/1-resnet152rcbam-3b-3cls-f12/model_epoch_best.pth' --name 3b-rcbam-f12-ukb-3cls --dataroot /root/ZYZ/GRINLAB/UKB-test --test_threshold 0.5 --testsets test --isRecord --mode 3cls
# python test_ukb_eval.py --model_name 3branch-cbam --model_path 'checkpoints/1-resnet152cbam-3b-3cls/model_epoch_best.pth' --name 3b-cbam-ukb-3cls --dataroot /root/ZYZ/GRINLAB/UKB-test --test_threshold 0.5 --testsets test --isRecord --mode 3cls

# python test_ukb_eval.py --model_name 3branch-rcbam --model_path 'checkpoints/1-resnet152rcbam-3b-f12/model_epoch_best.pth' --name 3b-rcbam-f12-ukb --dataroot /root/ZYZ/GRINLAB/UKB-test --test_threshold 0.5 --testsets test --isRecord



# python eval.py --model_name 3branch --model_path 'checkpoints/1_resnet152/model_epoch_best.pth' --name 3b-test-best --test_threshold 0.5 --testsets test --isRecord 
# python eval_v2.py --model_name 3branch-cbam --model_path 'checkpoints/1-resnet152cbam-3b-3cls/model_epoch_latest.pth' --name 3b-cbam-test-latest --test_threshold 0.5 --testsets test --isRecord --mode 3cls

# python eval_v2.py --model_name 3branch-cbam --model_path 'checkpoints/1-resnet152cbam-3b-3cls/model_epoch_best.pth' --name 3b-cbam-smdg-3cls --dataroot /root/ZYZ/GRINLAB/SMDG_test --test_threshold 0.5 --testsets test --isRecord
python eval_v2.py --model_name 3branch-rcbam --model_path 'checkpoints/1-resnet152rcbam-3b-f12/model_epoch_best.pth' --name 3b-rcbam-smdg --dataroot /root/ZYZ/GRINLAB/SMDG_test --test_threshold 0.5 --testsets test --isRecord
