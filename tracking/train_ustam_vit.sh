# There are the detailed training settings for MixFormer-ViT-B and MixFormer-ViT-L.
# 1. download pretrained ViT-MAE models (mae_pretrain_vit_base.pth.pth/mae_pretrain_vit_large.pth) at https://github.com/facebookresearch/mae
# 2. set the proper pretrained CvT models path 'MODEL:BACKBONE:PRETRAINED_PATH' at experiment/ustam_vit/CONFIG_NAME.yaml.
# 3. uncomment the following code to train corresponding trackers.

### Training USTAM-ViT-B
python tracking/train.py --script ustam_vit --config baseline_got --save_dir output --mode multiple --nproc_per_node 4


### Training USTAM-L
#python tracking/train.py --script ustam_vit --config baseline_large --save_dir output --mode multiple --nproc_per_node 4


### Training USTAM-B_GOT
#python tracking/train.py --script ustam_vit --config baseline_got --save_dir /YOUR/PATH/TO/SAVE/USTAM_GOT --mode multiple --nproc_per_node 8
