# Different test settings for USTAM-ViT-b, USTAM-ViT-l on LaSOT/TrackingNet/GOT10K/UAV123/OTB100
# First, put your trained USTAM models on SAVE_DIR/models directory. 
# Then,uncomment the code of corresponding test settings.
# Finally, you can find the tracking results on RESULTS_PATH and the tracking plots on RESULTS_PLOT_PATH.

##########-------------- USTAM-B -----------------##########
### LaSOT test and evaluation
# python tracking/test.py ustam_vit baseline_large --dataset lasot --threads 4 --num_gpus 2 --params__model USTAM_L_S/USTAM_ep0480.pth.tar --params__search_area_scale 5.05
# python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline_large

# ### TrackingNet test and pack
# python tracking/test.py ustam_vit baseline_large --dataset trackingnet --threads 2 --num_gpus 1 --params__model USTAM_L_S/USTAM_ep0480.pth.tar
# python lib/test/utils/transform_trackingnet.py --tracker_name ustam_vit --cfg_name baseline_large

### GOT10k test and pack
# python tracking/test.py ustam_vit baseline --dataset got_10k_test --threads 3 --num_gpus 1 --params__model spatial/USTAM_ep0450.pth.tar
# python lib/test/utils/transform_got10k.py --tracker_name ustam_vit --cfg_name baseline


