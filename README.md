# MGCNTrans

This is the code of our paper [Multiple-Frames Lifting Graph Convolution Transformer for 3D Human Pose Estimation](). This repository is based on the [JointFormer](https://github.com/seblutz/JointFormer) repository. Please refer to their readme or to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to setup the training data.

## Installation
python 3.7.0  
pytorch 1.8.1  
numpy 1.21.5  
opencv-python 4.5.5.64  
mayavi 4.7.4  
MATLAB 2020a  

## Dataset setup

### Human3.6M
Please download the dataset from [Human3.6M](http://vision.imar.ro/human3.6m/) website and refer to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to set up the Human3.6M dataset ('./dataset' directory). Or you can download the processed data from [here](https://drive.google.com/drive/folders/1_21m_TzMK8-o0n3UYoEW23W8HyPWyOTU?usp=sharing). 

### MPI-INF-3DHP
We set up the MPI-INF-3DHP dataset by ourselves. We convert the original data in `.mat` format to the processed data in `.npz` format by using `data_to_npz_3dhp.py` and `data_to_npz_3dhp_test.py`. You can download the processed data from [here](https://drive.google.com/drive/folders/1_21m_TzMK8-o0n3UYoEW23W8HyPWyOTU?usp=sharing). Put them in the `./dataset` directory. In addition, if you want to get the PCK and AUC metrics on this dataset, you also need to download the original dataset from the [official website](https://vcai.mpi-inf.mpg.de/3dhp-dataset/). After downloading the dataset, you can place the `TS1` to `TS6` folders in the test set under the `./3dhp_test` folder in this repo. 

```bash
${MGCNTrans}/
|-- data
|   |-- data_3d_h36m.npz
|   |-- data_2d_h36m_gt.npz
|   |-- data_2d_h36m_cpn_ft_h36m_dbb.npz
|   |-- data_test_3dhp.npz
|   |-- data_train_3dhp.npz
```

## Trained from scratch and Evaluating our models
### Human3.6M
Trained weights for the MGCNTrans on Human3.6M dataset can be found [here](https://drive.google.com/drive/folders/1t1CF6XNFJoHMNUq181-VosDSvBTPIMbW?usp=sharing). Put pre-training weights under 'checkpoint'. The parameters we used to train these weights were:
```
python run_mgcntrans_h36m.py --batch_size 256 --num_workers 2 --epochs 60 --keypoints gt --hid_dim 256 --intermediate --pred_dropout 0.2 --augment  
```

### MPI-INF-3DHP
Trained weights for the MGCNTrans on MPI-INF-3DHP dataset can be found [here](https://drive.google.com/drive/folders/1t1CF6XNFJoHMNUq181-VosDSvBTPIMbW?usp=sharing). The parameters we used to train these weights were:
```
python run_mgcntrans_3dhp.py --batch_size 256 --num_workers 2 --epochs 60 --keypoints gt --hid_dim 256 --intermediate --pred_dropout 0.2 --augment  
```
To evaluate our trained weights and generate the evaluation values in the tables of the paper, please append `--evaluate {path/to/weights}` to the respective training commands.
```
After testing on the MPI-INF-3DHP dataset, an 'inference_data.mat' file is generated in the 'results' folder, which can be tested AUC and PCK using the mpii_test_predictions_py.m file in the '3dhp_test/test_util' folder.
The result is 'mpii_3dhp_evaluation_activitywise.csv' in the '3dhp_test' folder.
```

## visualization
Visualization can be done using 'vis.py' in the 'demo' folder.   Modify '--pre' in 'vis.py' to be the training weight address.

First, you need to download YOLOv3 and HRNet pretrained models [here](https://drive.google.com/drive/folders/1twGLYjw0pbSbFjGYw22p5De0qygZ0yc0?usp=sharing) and put it in the './demo/lib/checkpoint' directory. 

Then, you need to put your videos in the './demo/video' directory. The results will be saved under './demo/demo' directory.


## Acknowledgements
Our code is extended from the following repositories. We thank their authors for releasing their code.
*[JointFormer](https://github.com/seblutz/JointFormer)
*[GraphMLP](https://github.com/vegetebird/graphmlp)
*[VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
*[P-STMO](https://github.com/patrick-swk/p-stmo)
*[SemGCN](https://github.com/garyzhao/SemGCN)