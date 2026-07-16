# MGCNTrans: Multiple-Frames Lifting Graph Convolution Transformer for 3D Human Pose Estimation

This repository provides a PyTorch implementation of **MGCNTrans**, a multiple-frame lifting framework for 3D human pose estimation. MGCNTrans combines temporal self-attention with spatial graph convolution to capture motion dynamics and human-joint relationships. The repository includes code for model training, evaluation, and visualization on the Human3.6M and MPI-INF-3DHP datasets.

## Framework Overview

![Overview of the MGCNTrans framework](fig/Overview1.png)

The overall framework of MGCNTrans is shown above. Given a sequence of images, a 2D pose estimator first extracts a sequence of 2D poses. Feature embeddings and positional encodings are then used to construct joint tokens, which are processed by stacked STGraphFormer blocks. Each block combines temporal multi-head self-attention (MHSA) with a spatial MHSA+TGCN branch to capture temporal dependencies and spatial relationships among human joints. In the spatial branch, the TGCN-MLP module further refines the joint representations through spatial and channel graph convolutions. Finally, a regression head lifts the refined representations to a 3D human pose sequence.

## Qualitative Comparison

![Qualitative comparison on the Human3.6M dataset](fig/five_groups_data.png)

Qualitative comparison on the **Directions** action from the Human3.6M dataset. From left to right, the columns show the input image, ground truth, DDHPose, PRPose, TCPFormer, and MGCNTrans (Ours). The orange arrows highlight differences in the estimated joint positions. MGCNTrans produces more accurate upper-limb poses and better preserves bone-length consistency and natural motion patterns, resulting in more faithful and structurally coherent 3D pose estimates.

## Installation
python 3.7.0  
pytorch 1.8.1  
numpy 1.21.5  
opencv-python 4.5.5.64  
mayavi 4.7.4  
MATLAB 2020a  

## Dataset setup

### Human3.6M
Please download the dataset from [Human3.6M](http://vision.imar.ro/human3.6m/) website and refer to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to set up the Human3.6M dataset (`./data` directory). Or you can download the processed data from [here](https://drive.google.com/drive/folders/1E82Kwtjjvh4pnaCShsymI06pqJ76Hqru). 

### MPI-INF-3DHP
We set up the MPI-INF-3DHP dataset by ourselves. We convert the original data in `.mat` format to the processed data in `.npz` format by using `data_to_npz_3dhp.py` and `data_to_npz_3dhp_test.py`. You can download the processed data from [here](https://drive.google.com/drive/folders/1E82Kwtjjvh4pnaCShsymI06pqJ76Hqru). Put them in the `./data` directory. In addition, if you want to get the PCK and AUC metrics on this dataset, you also need to download the original dataset from the [official website](https://vcai.mpi-inf.mpg.de/3dhp-dataset/). After downloading the dataset, you can place the `TS1` to `TS6` folders in the test set under the `./3dhp_test` folder in this repo. 

```bash
${MGCNTrans}/
|-- data
|   |-- data_3d_h36m.npz
|   |-- data_2d_h36m_gt.npz
|   |-- data_2d_h36m_cpn_ft_h36m_dbb.npz
|   |-- data_test_3dhp.npz
|   |-- data_train_3dhp.npz
```

## Training from Scratch and Evaluation
### Human3.6M
To train MGCNTrans from scratch on the Human3.6M dataset, use the following command with the specified training parameters:
```
python run_mgcntrans_h36m.py --batch_size 256 --num_workers 2 --epochs 60 --keypoints gt --hid_dim 256 --intermediate --pred_dropout 0.2 --augment  
```

### MPI-INF-3DHP
To train MGCNTrans from scratch on the MPI-INF-3DHP dataset, use the following command with the specified training parameters:
```
python run_mgcntrans_3dhp.py --batch_size 256 --num_workers 2 --epochs 60 --keypoints gt --hid_dim 256 --intermediate --pred_dropout 0.2 --augment  
```
To evaluate a trained model, append `--evaluate {path/to/checkpoint}` to the corresponding command.
```
After testing on the MPI-INF-3DHP dataset, an 'inference_data.mat' file is generated in the 'results' folder, which can be tested AUC and PCK using the mpii_test_predictions_py.m file in the '3dhp_test/test_util' folder.
The result is 'mpii_3dhp_evaluation_activitywise.csv' in the '3dhp_test' folder.
```

## Visualization
Visualization can be performed using `demo/vis.py`. Set the `--pre` argument in `vis.py` to the path of the trained model checkpoint.

First, you need to download YOLOv3 and HRNet pretrained models [here](https://drive.google.com/drive/folders/1twGLYjw0pbSbFjGYw22p5De0qygZ0yc0?usp=sharing) and put it in the './demo/lib/checkpoint' directory. 

Then, you need to put your videos in the './demo/video' directory. The results will be saved under './demo/demo' directory.


## Acknowledgements
Our code is extended from the following repositories. We thank their authors for releasing their code.
*[JointFormer](https://github.com/seblutz/JointFormer)
*[GraphMLP](https://github.com/vegetebird/graphmlp)
*[VideoPose3D](https://github.com/facebookresearch/VideoPose3D)
*[P-STMO](https://github.com/patrick-swk/p-stmo)
*[SemGCN](https://github.com/garyzhao/SemGCN)
