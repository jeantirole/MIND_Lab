

def ericFunction( input_ ):
    print(input_)


#------------- GET Model Inference --------------------------------------------------#

def infer_model(input_file_path, thres,gpu_id):
	import os
	os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
	from mmdet.apis import init_detector, show_result_pyplot
	from mmrotate.apis import inference_detector_by_patches
	import mmrotate
	import torch
	import cv2
	import numpy as np
	#print(mmrotate)
	#mmrotate.__version__=='0.3.4'


	#------ image part 

	# print("load image")
	# img_root = "./post_images/"
	# #img_id = "black_sky_result_roitrans_5_epoch_thr_0.75.png"
	# img_id = "testfile.png"
	# whole_png = os.path.join(img_root,img_id)

	# whole_png= "/mnt/hdd/eric/.tmp_ipy/00.Reproduction_Test/post_images/testfile.PNG"

	whole_png = input_file_path
	
	img = cv2.imread(whole_png)
	#from PIL import Image
	#Image.MAX_IMAGE_PIXELS = None
	#img = Image.open(whole_png)
	#img = np.array(img)
	#-----
	choose_epoch = 9
	checkpoint_file = f"/mnt/hdd/eric/.tmp_ipy/00.Checkpoint/ship_tmp/epoch_{choose_epoch}.pth"
	#checkpoint_file = "/mnt/hdd/eric/.tmp_ipy/00.Checkpoint/ship_tmp_multi/epoch_15.pth"
	config_file = '/mnt/hdd/eric/.tmp_ipy/00.Checkpoint/KARI_tmp/kari_fine_tunning.py'
	#config_file = "/mnt/hdd/eric/.tmp_ipy/00.Checkpoint/ship_tmp_multi/ship_fine_tunning.py"

	from mmcv import Config
	cfg = Config.fromfile(config_file)

	#----- classes
	cfg.model.roi_head.bbox_head[0].num_classes=1 
	cfg.model.roi_head.bbox_head[1].num_classes=1

	classes = ('ship',)
	# classes=(
    #         'Other_Ship', 'Other_Warship', 'Submarine',
    #         'Other_Aircraft_Carrier', 'Ticonderoga', 'Other_Destroyer',
    #         'Other_Frigate', 'Patrol', 'Other_Landing', 'Commander',
    #         'Other_Auxiliary_Ship', 'Other_Merchant', 'Container_Ship', 'RoRo',
    #         'Cargo', 'Barge', 'Tugboat', 'Ferry', 'Yacht', 'Sailboat',
    #         'Fishing_Vessel', 'Oil_Tanker', 'Hovercraft', 'Motorboat', 'Dock',
    #     )
	
	
	cfg.data.train.classes=classes
	cfg.data.val.classes=classes
	cfg.data.test.classes=classes

	# batch 
	cfg.data.samples_per_gpu = 1

	cfg.gpu_ids = gpu_id
	cfg.device='cuda'
	cfg.seed=22

	# random seed
	from mmdet.apis import set_random_seed
	set_random_seed(cfg.seed,deterministic=True)

	model = init_detector( cfg, checkpoint_file,device=f'cuda:{cfg.gpu_ids}')
	print("#------------------- init detection")
	from easydict import EasyDict

	args = EasyDict()
	args.img = img
	# args.patch_sizes = [1024,2048]
	# args.patch_steps = [412,824]
	# args.img_ratios = [1.0,1.5,2.0]

	# default
	args.patch_sizes = [1024,800,2048]
	args.patch_steps = [824,672,1920]
	args.img_ratios = [1.0,1.5,2.0]

	args.merge_iou_thr = 0.1

	args.score_thr = thres
	args.palette = 'dota'


	result = inference_detector_by_patches( model, 
                                            args.img, 
                                            args.patch_sizes,
											args.patch_steps, 
                                        	args.img_ratios,
											args.merge_iou_thr
                                           )

	output_file_name = [i for i in whole_png.split("/")][-1] + "_result"

	# show_result_pyplot(
	# model,
	# args.img,
	# result,
	# palette=args.palette,
	# score_thr=args.score_thr,
	# out_file=os.path.join("/mnt/hdd/eric/.tmp_ipy/00.Reproduction_Test/[Ship]_AIS_Matching_Result", output_file_name) )

	return result