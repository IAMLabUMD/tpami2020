# Models built on Fully Convolutional Networks
This repository builds upon an existing repository that uses the Fully Convolutional Networks: forked from [MarvinTeichmann/KittiSeg](https://github.com/MarvinTeichmann/KittiSeg).

The code was modified and updated, using Python 3.6 and Tensorflow 1.9.

# Installation
## Environment setup
1. Clone or download this repository.
2. Change directory to the current folder.
  ```
  cd Hand-Object-Models
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
5. Install all submodules.
  ```
  git submodule update --init --recursive
  ```

## Dataset download
1. Download TEgO dataset from [this link](https://drive.google.com/file/d/18CuMGlRzkmN9rouzWA-eql3-Wm1eLw7K/view?usp=sharing) and place it in the `DATA` folder.
2. Untar `TEgO_with_VOC.tar.gz` in the `data` folder.

TODOs:
- [ ] GTEA
- [ ] GTEA Gaze+


## Pre-trained models
1. Download a hand segmentation model from [this link](https://drive.google.com/file/d/1YpOghS_KGya_4yasO9ejLAudE8cyq3Fl/view?usp=sharing) and place it in the `RUNS` folder.
2. Untar `fcn_hand_model.tar.gz` in the `RUNS` folder.


# Only Object Center Estimation Models

## Fine-tuned model
The hand segmentation is fine-tuned to perform object center estimation. Note that each image has up to one object of interest, which center area is annotated with a heatmap blob.

* Training
```
python train_local.py --hypes hypes/local_finetune_TEgO.json
```


## Multi-class model
A multi-class approach is used to perform both hand segmentation and object center estimation.

* Training
```
python train_local.py --hypes hypes/local_multiclass_TEgO.json
```


## Multi-task model
A multi-task approach is used to perform both hand segmentation and object center estimation.

* Training
```
python train_local.py --hypes hypes/local_multitask_TEgO.json
```


## Hand-primed model
The hand-primed model consists of two fully convolutional networks---one for hand segmentation and the other for object center estimation. The output of the hand segmentation model is infused into the object center estimation model.

* Training
```
python train_local.py --hypes hypes/local_handprimed_TEgO.json
```

## Model evaluation with TEgO
```
python evaluate_local.py --model [the path of a model being evaluated] \
                         --input_file DATA/test_TEgO_local.txt \
                         --model_type [finetune|multitask|multiclass|handprimed]
```

Note: additional parameters for the evaluation script: 
- --threshold [a threshold value for a model's estimation]
- --width [image width]
- --height [image height]


# Object Center Esitmation & Object Recognition Models


## Fine-tuned model
* Training
```
python train_recog.py --hypes hypes/recog_finetune_TEgO.json
```


## Multi-class model
* Training
```
python train_recog.py --hypes hypes/recog_multiclass_TEgO.json
```


## Multi-task model
* Training
```
python train_recog.py --hypes hypes/recog_multitask_TEgO.json
```


## Hand-primed model
* Training
```
python train_recog.py --hypes hypes/recog_handprimed_TEgO.json
```


## Model evaluation with TEgO
```
python evaluate_recog.py --model [the path of a model being evaluated] \
                         --input_file DATA/test_TEgO_local.txt \
                         --model_type [finetune|multitask|multiclass|handprimed]
```

Note: additional parameters for the evaluation script: 
- --threshold [a threshold value for a model's estimation]
- --width [image width]
- --height [image height]


# Contact
Please contact us at iamlab@umd.edu if you find any issues.