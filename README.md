# 3D Asymmetric Convolutional Segmentation Network (3D-ACSNet) for Brain MR Images of 6-month-old Infants

This repository provides the experimental code for our paper "3D Asymmetric Convolutional Segmentation Network (3D-ACSNet) for Brain MR Images of 6-month-old Infants".

Created by Zilong Zeng at Beijing Normal University. **For any questions, please contact ricardo_zeng@mail.bnu.edu.cn or tengdazhao@bnu.edu.cn**

![framework]("/images/framework.png"):

## Contents
  - [Publication](#publication)
  - [Requirements](#requirements)
  - [Dataset](#dataset)
  - [Runing the code](#runing-the-code)
    - [Training](#training)
    - [Model Fusion](#model-fusion)
    - [Testing](#testing)
  - [Results](#results)

## Publication
If you find that this work is useful for your research, please consider citing our paper.

## Requirements
- python>=3.5
- tensorflow-gpu==1.14.0
- Keras==2.2.5
- nibabel>=2.4.1

## Dataset
The dataset used for model training and validation is from [iSeg-2019](http://iseg2019.web.unc.edu/). The iSeg organizers provide 10 infant subjects with labels for model training and validation, and 13 infant subjects without labels for model testing. Each subject consists of T1 and T2 images for segmentation.

## Runing the code
The process of **training**, **model fusion** and **testing** are all completed by configuring JSON files.

### Training
An example of JSON file for the training process shown in [train.json](/settings/train.json).
  - "task": This parameter need to set to `"train"`, which means that we're going to train the model.
  - "name": The name of this training process.
  - "gpu": For example, if you have 3 gpus, but you want to use the 1st and 3rd gpu, you should write here as `"0,2"`
  - "cube_size": The size of extracted cubes during training process.
  - "save_period": Model save interval.
  - "train_dir": The path to your training dataset.
  - "train_ids": The ids of your training subjects.
  - "batch_size": Batch size.
  - "cycle": The training epochs in one cycle in the cosine annealing learning schedule.
  - "lr_init": The maximum learning rate in each cycle.
  - "lr_min": The minimum learning rate in each cycle.
  - "epochs": The total training epochs.
  - "train_nums_per_epoch": The total training cubes extracted in one epoch.
  - "load_model_file": The path to pretrain model to be loaded. If not model file to be loade, you should set here as `null`.

Once you have configured your JSON file for training, you need run this command like this:
```
python -m client.run ./settings/train.json
```

### Model Fusion

Once your training process is over, you could configure your JSON file to fuse your model's parameters in **BN fusion** and **Branch Fusion** process. An example of JSON file for the model fusion process shown in [deploy.json](/settings/deploy.json).
  - "task": This parameter need to set to `"deploy"`, which means that we're going to fuse the model's parameters.
  - "ori_model": List of pathes to model which has not fused parameters.
  - "save_fold": The path to folder where fused model will be stored.

Once you have configured your JSON file for model fusion, you need run this command like this:
```
python -m client.run ./settings/deploy.json
```

### Testing
Once your training or model fusion process is over, you can configure your JSON file to segment subjects' brain images. An example of JSON file for the model prediction shown in [predict_undeploy.json](/settings/predict_undeploy.json).
  - "task": This parameter need to set to `"predict"`, which means that we're going to make model prediction.
  - "gpu_id": The id of the GPU which you want to use. For example, you want to use the second gpu, you should write `"1"`.
  - "save_folder": The path to the folder of the saved segmentation results.
  - "data_path": The path to the images to be segmented.
    - Notice!! If you want to use a different dataset here with T1 and T2 images, you dataset should be organized like this:
      ```
      ├── subject-1-label.hdr
      ├── subject-1-label.img
      ├── subject-1-T1.hdr
      ├── subject-1-T1.img
      ├── subject-1-T2.hdr
      ├── subject-1-T2.img
      ├── subject-2-label.hdr
      ├── subject-2-label.img
      ├── subject-2-T1.hdr
      ├── subject-2-T1.img
      ├── subject-2-T2.hdr
      ├── subject-2-T2.img
      ├── ...
      ```
  - "cube_size": The size of extracted cubes during prediction process.
  - "crop_size": The overlapping step size. A small step size lead to better result.
  - "subjects": The list of ids of subjects to be segmented.
  - "predict_mode": two optional choice —— `"evaluation"` and `"prediction"`
    - `"evaluation"`: If you have labels, you can set this option and evaluate the model's accuracy.
    - `"prediction"`: If you do not have labels, you need to set this.
  - "model_files": The list of model files to be loaded for model prediction.
    - If there is only one model file's path in this parameter, the program will output one segmentation result predicted by this model file. See examples in [predict_deploy.json](/settings/predict_deploy.json) or [predict_undeploy.json](/settings/predict_undeploy.json)
    - If there are multiple model files' pathes in this parameter, the program will adopt the majority voting strategy to combine these models' segmentation results. See example in [ensemble_deploy.json](/settings/ensemble_deploy.json) or [ensemble_undeploy.json](/settings/ensemble_undeploy.json)
  - "deploy": If the model file to be loaded has fused parameters, you should set this parameter as `true`; otherwise, you need to set here as `false`.
  - "label_path": The path to the folder which contains the segmentation labels.

  We provide the pretrained models which are used for ***the validation dataset in*** iSeg-2019 competition.
  - You can run command like this:
      ```
      python -m client.run ./settings/ensemble_deploy.json
      ```
      or
      ```
      python -m client.run ./settings/ensemble_undeploy.json
      ```
      You should achieve the result as the table:
      |  CSF       | GM          | WM    | Average 
      |:----------:|:-----------:|:-----:|:--------------:|
      | 96.0700 | 92.8933 | 90.7167 | 93.2267 |

  - We also release the quantitative evaluation results: `evaluation_result_sw_bnu0403.xlsx`.

## Results
**Comparison of segmentation performance on the 13 validation infants of iSeg-2019 between the proposed method and the methods of the top 4 ranked teams.**
- DICE    
  |Team   |  CSF       | GM          | WM    | Average 
  |:----------:|:----------:|:-----------:|:-----:|:--------------:|
  |Brain_Tech|0.961|0.928|0.911|0.933|
  |FightAutism|0.960|0.929|0.911|0.933|
  |OxfordIBME|0.960|0.927|0.907|0.931|
  |QL111111|0.959|0.926|0.908|0.931|
  |Our|**0.961**|**0.930**|**0.911**|**0.934**|
- MHD
  |Team   |  CSF       | GM          | WM    | Average 
  |:----------:|:----------:|:-----------:|:-----:|:--------------:|
  |Brain_Tech|8.873|5.724|7.114|7.237|
  |FightAutism|9.233|5.678|**6.678**|7.196|
  |OxfordIBME|**8.560**|**5.495**|6.759|**6.938**|
  |QL111111|9.484|5.601|7.028|7.371|
  |Our|8.920|5.781|7.165|7.287|
- ASD
  |Team   |  CSF       | GM          | WM    | Average 
  |:----------:|:----------:|:-----------:|:-----:|:--------------:|
  |Brain_Tech|0.108|0.300|0.347|0.252|
  |FightAutism|0.110|0.300|0.341|0.250|
  |OxfordIBME|0.112|0.307|0.353|0.257|
  |QL111111|0.114|0.307|0.353|0.258|
  |Our|**0.107**|**0.295**|**0.337**|**0.246**|

