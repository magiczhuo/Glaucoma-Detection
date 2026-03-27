## 1. Pre-segmentation of ROI images:
The pre-trained segmentation model is in './AutoRetinalImageSegmentation'. **Be careful that the python environment is different from the one used for classification**. Please set up two virtual environments for segmentation and classification respectively. 
```
cd .
conda env create --file environment.yaml
cd ./AutoRetinalImageSegmentation
conda env create --file environment.yaml
```

Run `./AutoRetinalImageSegmentation/roi_seg.py` for segmentation. You can define the ROI view size flexibly. Our current model is trained with a higher view (ROI 800) because the segmentation model is not that accurate. If you get a better segmentation model, we suggest to use a lower view. 

We also use CLAHE for image enhancement for ROI images. 

## 2. Dataset
### Training and validation dataset
The dataset should be arranged as this:
	
	Training and validation dataset
		|- train(90% of downloaded dataset)
			|_ 0_neg # save the full images
				xxxx.png
				...
			|_ 1_pos
				yyyy.png
				...
			|_ 0_roi # save the ROI images
				xxxx.png
				...
			|_ 1_roi
				yyyy.png
				...

		|- val(10% of downloaded dataset)
			|_ 0_neg 
				xxxx.png
				...
			|_ 1_pos
				yyyy.png
				...
			|_ 0_roi
				xxxx.png
				...
			|_ 1_roi
				yyyy.png
				...
### Testing dataset
Before testing the model, please arrange the testing dataset as following:
	
	Testing dataset
		|- test1
			|_ 0_neg
				xxxx.png
				...
			|_ 1_pos
				yyyy.png
				...
			|_ 0_roi
				xxxx.png
				...
			|_ 1_roi
				yyyy.png
				...
		|- test2
			|_ 0_neg
				xxxx.png
				...
			|_ 1_pos
				yyyy.png
				...
			|_ 0_roi
				xxxx.png
				...
			|_ 1_roi
				yyyy.png
				...



## Train the model

We provide an example script to train our model by running `bash train.sh`, in which you can change the following parameters:

`--name`: the directory name you want to save your checkpoints in.

`--data_aug`: whether or not use data augmentations, including RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, GaussianBlur

`--aug_prob`: the augmentation probability

`--model_name`: the backbone model to use. '3branch-cbam': 3 branches with ResNet152 + CBAM; '2branch-cbam': 2 branches with ResNet152 + CBAM, no dynamic window; '3branch': 3 branches with ResNet50. 
  
`--dataroot`: path of training and validation datasets.

`--continue_train`: continue training from last checkpoint. 

`--epoch_count`: use with `--continue_train`, the last epoch recorded.
  
## Test the model
			
We provide an example script to test our model by running `bash test.sh`. 

## Best pretrained model
The current best checkpoint is:

`./checkpoints/1-resnet152cbam-2b/model_epoch_best.pth`