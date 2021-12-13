# tf-faster-rcnn
**This repository explains how to train & test your own data on the Faster-RCNN model.**

# Environment setup list:
1. Python 3.6
2. CUDA 10.0
3. cuDNN 7.4.
4. tensorflow_gpu-1.14.0
5. Visual Studio C++ Build Tools 2015

# How to train on your own dataset?
1. To use this model, you need to change your data to the [VOC2007-like dataset](https://www.programmersought.com/article/65711056356/;jsessionid=421CF30D7DDB52E78C87ABD7477A08E3).

2. Change the label classes on line 33/34 of ```lib/datasets/pascal_voc.py``` (also ```demo.py``` in root folder) to your own classes (**DO NOT DELETE BACKGROUND CLASS**).

3. (Optional) You can change the ```learning rate```, ```max_iters```, etc in ```lib/config/config.py```. However, ```snap_iterations``` need to be **SMALLER** than ```max_iters```.

4. Change ```NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)} DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}```  in ```demo.py``` to  ```NETS = {'vgg16': ('vgg16.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)} DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval',)}```.

5. Create ```output/vgg16/voc_2007_trainval/default``` folder under the root folder.

6. Run ```train.py``` to start training. 

7. After training, the model will be in ```default/voc_2007_trainval/default```. Copy & rename them to the ```output/vgg16/voc_2007_trainval/default``` folder. 

![Renamed Files](/20191102010907813.jpg)

8. To re-train the model, make sure you delete the files under ```default/voc_2007_trainval/default```, ```output/vgg16/voc_2007_trainval/default```, and ```data/cache```.


# How to test your trained model?
1. Put the images you want to test under ```data\demo```.

2. Rename the ```im_names``` in ```demo.py``` to the names of the images you want to test.

3. Change ```default='res101'``` on in ```demo.py``` to ```default='vgg16'```.

4. Run demo.py.

<br>
<br>
**Below is the same as the original repository.**

# How To Use This Branch
1. Install tensorflow, preferably GPU version. Follow [instructions]( https://www.tensorflow.org/install/install_windows). If you do not install GPU version, you need to comment out all the GPU calls inside code and replace them with relavent CPU ones.

2. Checkout this branch

3. Install python packages (cython, python-opencv, easydict) by running  
`pip install -r requirements.txt`   
(if you are using an environment manager system such as `conda` you should follow its instruction)

4. Go to  ./data/coco/PythonAPI  
Run `python setup.py build_ext --inplace`  
Run `python setup.py build_ext install`  
Go to ./lib/utils and run `python setup.py build_ext --inplace`

5. Follow [these instructions](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to download PyCoco database.
I will be glad if you can contribute with a batch script to automatically download and fetch. The final structure has to look like  
`data\VOCDevkit2007\VOC2007`  

1. Download pre-trained VGG16 from [here](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) and place it as `data\imagenet_weights\vgg16.ckpt`.  
For rest of the models, please check [here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)

7. Run train.py

Notify me if there is any issue found.

