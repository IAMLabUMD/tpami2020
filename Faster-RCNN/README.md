# Models built on Faster R-CNN
This repository builds upon an existing repository that uses the Faster-RCNN model: forked from [dBeker/Faster-RCNN-TensorFlow-Python3](https://github.com/vincent317/Faster-RCNN-TensorFlow-Python3).

The code was built using Python 3.7 and Tensorflow 1.14.


# Installation
## Environment setup
1. Clone or download this repository.
2. Change directory to the current folder.
  ```
  cd Faster-RCNN
  ```
3. Install [pipenv](https://pipenv.pypa.io/en/latest/install/) if not installed.
4. Activate a virtual environment using pipenv.
  ```
  pipenv install
  ```
  *(Run the command below if the virtual environment with pipenv is already installed.)*
  ```
  pipenv shell
  ```
5. Install COCO Python API dependencies.
  ```
  cd data/coco/PythonAPI  
  python setup.py build_ext --inplace
  python setup.py build_ext install 
  ```
6. Install other libraries.
  ```
  cd lib/utils
  python setup.py build_ext --inplace
  ```
7. Download the pre-trained VGG16 from [this link](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) and place it as `data/imagenet_weights/vgg16.ckpt`.


## Dataset download
1. Download TEgO dataset from [this link](https://drive.google.com/file/d/18CuMGlRzkmN9rouzWA-eql3-Wm1eLw7K/view?usp=sharing) and place it in the `data` folder.
2. Untar `TEgO_with_VOC.tar.gz` in the `data` folder.


# Training & Testing
## TEgO with bounding boxes for object center areas
A bounding box surrounding an object center area is used to train and test a Faster R-CNN model. Note that each image has up to one object of interest, which center area is annotated with a bounding box.

* Training
```
python train.py --dataset tego
```

* Testing
```
python evaluate.py tego
```

## TEgO with bounding boxes for whole objects
A bounding box surrounding a whole object is used to train andn test a Faster R-CNN model. Note that each image has up to one object of interest, which is annotated with a bounding box.
* Training
```
python train.py --dataset tego_wholeBB
```

* Testing
```
python evaluate.py tego_wholeBB
```

## TEgO Blind with bounding boxes for object center areas
Data collected by the blind individual are used. A bounding box surrounding an object center area is used to train and test a Faster R-CNN model. Note that each image has up to one object of interest, which center area is annotated with a bounding box.

* Training
```
python train.py --dataset tego_blind
```

* Testing
```
python evaluate.py tego_blind
```

## TEgO Blind with bounding boxes for whole objects
Data collected by the blind individual are used. A bounding box surrounding a whole object is used to train andn test a Faster R-CNN model. Note that each image has up to one object of interest, which is annotated with a bounding box.

* Training
```
python train.py --dataset tego_blind_wholeBB
```

* Testing
```
python evaluate.py tego_blind_wholeBB
```

## TEgO Sighted with bounding boxes for object center areas
Data collected by the sighted individual are used. A bounding box surrounding an object center area is used to train and test a Faster R-CNN model. Note that each image has up to one object of interest, which center area is annotated with a bounding box.

* Training
```
python train.py --dataset tego_sighted
```

* Testing
```
python evaluate.py tego_sighted
```

## TEgO Sighted with bounding boxes for whole objects
Data collected by the sighted individual are used. A bounding box surrounding a whole object is used to train andn test a Faster R-CNN model. Note that each image has up to one object of interest, which is annotated with a bounding box.

* Training
```
python train.py --dataset tego_sighted_wholeBB
```

* Testing
```
python evaluate.py tego_sighted_wholeBB
```

# Contact
Please contact us at iamlab@umd.edu if you find any issues.