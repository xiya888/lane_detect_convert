# lane_detect_convert

Ultra-Fast-Lane-Detection  

PyTorch implementation of the paper "Ultra Fast Structure-aware Deep Lane Detection".  

Operate Demo
---
   >本说明是对车道线检测算法Ultra-Fast-Lane-Detection(代码路径：https://github.com/cfzd/Ultra-Fast-Lane-Detection)的pytorch模型(代码根据实际需要有改动)转换成caffe模型。首先，要感谢所有原作者的创作和灵感，结合实际项目对车道线检测算法的模型转换做一个实践操作，经测试，是可以跑通的。
   >仅供科学研究，注意在使用过程中如涉及到知识产权问题时请标注来源并和原作者联系沟通，这里不对产权问题负责。
   >具体操作流程:
	>1、安装配置caffe，并编译好，见文件夹caffe/，编译参考其说明
	>2、模型转换，参考转换代码路径为：https://github.com/hanson-young/nniefacelib代码有修改，现同步到github上，思路是pytorch->onnx->caffemodel
	   >如何使用：
	   >1）存放车道线算法训练后的pytorch模型，如存放在：nniefacelib/PFPLD/models/pretrained/culane_18_download.pth
	   >2) 运行convert_to_onnx.py，生成onnx模型，具体路径可设置，如存放在：nniefacelib/PFPLD/models/onnx/culane_18_download.onnx
	   >3）生成caffemodel及模型参数文件，执行nniefacelib/PFPLD/cvtcaffe/convertCaffe.py，生成的文件可设置，如存放在nniefacelib/PFPLD/cvtcaffe/models/
	   >4）实际运行时只需在上述代码中修改下路径即可。
	>3、模型的推理测试
	   >代码路径，同样更新到github上，参考代码：https://github.com/Jade999/caffe_lane_detection，代码根据实际有改动
	   >如何使用：
	   >1）进入caffe_lane_detection，将1中转换的文件(.caffemodel和.prototxt)放到该文件夹下，执行inference.py
	>4、项目使用
	   >可根据2中推理的代码进行修改
    >说明：
	   >1、经过测试，目前转换后模型推理代码那块，对输入分辨率1640*590没有问题，但对于1280*720会有显示问题，需要再调整
	   

##Other readme:
------
附车道线检测算法Ultra-Fast-Lane-Detection部分说明：
	### Ultra-Fast-Lane-Detection
	PyTorch implementation of the paper "[Ultra Fast Structure-aware Deep Lane Detection](https://arxiv.org/abs/2004.11757)".

	Updates: Our paper has been accepted by ECCV2020.

	![alt text](vis.jpg "vis")

	The evaluation code is modified from [SCNN](https://github.com/XingangPan/SCNN) and [Tusimple Benchmark](https://github.com/TuSimple/tusimple-benchmark).

	Caffe model and prototxt can be found [here](https://github.com/Jade999/caffe_lane_detection).

	### Demo 
	<a href="http://www.youtube.com/watch?feature=player_embedded&v=lnFbAG3GBN4
	" target="_blank"><img src="http://img.youtube.com/vi/lnFbAG3GBN4/0.jpg" 
	alt="Demo" width="240" height="180" border="10" /></a>


	# Install
	Please see [INSTALL.md](./INSTALL.md)

	# Get started
	First of all, please modify `data_root` and `log_path` in your `configs/culane.py` or `configs/tusimple.py` config according to your environment. 
	- `data_root` is the path of your CULane dataset or Tusimple dataset. 
	- `log_path` is where tensorboard logs, trained models and code backup are stored. ***It should be placed outside of this project.***



	***

	For single gpu training, run
	```Shell
	python train.py configs/path_to_your_config
	```
	For multi-gpu training, run
	```Shell
	sh launch_training.sh
	```
	or
	```Shell
	python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py configs/path_to_your_config
	```
	If there is no pretrained torchvision model, multi-gpu training may result in multiple downloading. You can first download the corresponding models manually, and then restart the multi-gpu training.

	Since our code has auto backup function which will copy all codes to the `log_path` according to the gitignore, additional temp file might also be copied if it is not filtered by gitignore, which may block the execution if the temp files are large. So you should keep the working directory clean.
	***

	Besides config style settings, we also support command line style one. You can override a setting like
	```Shell
	python train.py configs/path_to_your_config --batch_size 8
	```
	The ```batch_size``` will be set to 8 during training.

	***

	To visualize the log with tensorboard, run

	```Shell
	tensorboard --logdir log_path --bind_all
	```

	# Trained models
	We provide two trained Res-18 models on CULane and Tusimple.

	|  Dataset | Metric paper | Metric This repo | Avg FPS on GTX 1080Ti |    Model    |
	|:--------:|:------------:|:----------------:|:-------------------:|:-----------:|
	| Tusimple |     95.87    |       95.82      |         306         | [GoogleDrive](https://drive.google.com/file/d/1WCYyur5ZaWczH15ecmeDowrW30xcLrCn/view?usp=sharing)/[BaiduDrive(code:bghd)](https://pan.baidu.com/s/1Fjm5yVq1JDpGjh4bdgdDLA) |
	|  CULane  |     68.4     |       69.7       |         324         | [GoogleDrive](https://drive.google.com/file/d/1zXBRTw50WOzvUp6XKsi8Zrk3MUC3uFuq/view?usp=sharing)/[BaiduDrive(code:w9tw)](https://pan.baidu.com/s/19Ig0TrV8MfmFTyCvbSa4ag) |

	For evaluation, run
	```Shell
	mkdir tmp
	# This a bad example, you should put the temp files outside the project.

	python test.py configs/culane.py --test_model path_to_culane_18.pth --test_work_dir ./tmp

	python test.py configs/tusimple.py --test_model path_to_tusimple_18.pth --test_work_dir ./tmp
	```

	Same as training, multi-gpu evaluation is also supported.

	# Visualization

	We provide a script to visualize the detection results. Run the following commands to visualize on the testing set of CULane and Tusimple.
	```Shell
	python demo.py configs/culane.py --test_model path_to_culane_18.pth
	# or
	python demo.py configs/tusimple.py --test_model path_to_tusimple_18.pth
	```

	Since the testing set of Tusimple is not ordered, the visualized video might look bad and we **do not recommend** doing this.

	# Speed
	To test the runtime, please run
	```Shell
	python speed_simple.py  
	# this will test the speed with a simple protocol and requires no additional dependencies

	python speed_real.py
	# this will test the speed with real video or camera input
	```
	It will loop 100 times and calculate the average runtime and fps in your environment.

	# Citation

	```
	@InProceedings{qin2020ultra,
	author = {Qin, Zequn and Wang, Huanyu and Li, Xi},
	title = {Ultra Fast Structure-aware Deep Lane Detection},
	booktitle = {The European Conference on Computer Vision (ECCV)},
	year = {2020}
	}
	```

	# Thanks
	Thanks zchrissirhcz for the contribution to the compile tool of CULane, KopiSoftware for contributing to the speed test, and ustclbh for testing on the Windows platform.
