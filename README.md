## StrongPose: Bottom-up, Strong Keypoints Heat Maps Base Person Pose Estimation.
This work established a StrongPose model which jointly tackle the problem of pose estimation and person detection. Towards this purpose we present a bottom-up box-free approach for the task of pose estimation and action recognition. The model utilizes a convolution network that learns how to detect Strong Keypoints Heat Maps (SKHM) and predict their comparative displacements, enabling us to group keypoints into person pose instances. Further, we produce Body Heat Maps (BHM) with the help of keypoints which allows us to localize human body in the picture. The StrongPose framework is based on a fully-convolutional engineering and permits proficient inference, with runtime basically autonomous of the number of individuals display within the scene. Train and test on COCO data alone, our framework achieves COCO test-dev keypoint average precision of 0.708 using ResNet-101 and 0.725 using ResNet-152, which considerably outperforms all prior bottom-up pose estimation frameworks.

Markup : * Bullet listDataset: COCO Keypoint 2017 <br/>
Markup : * Bullet list Platform: TensorFlow, OpenCV <br/>
Markup : * Bullet list Bacbone Network: ResNet 101, 152 (Modified) <br/>

## Introduction
Code repo for ICPR 2021 paper (https://Article-will-be-available-soon)

## Result
![](pic3.jpeg)

## Required
##### > Python3
##### > Tensorflow 1.80
##### > pycocotools 2.0
##### > skimage 0.13.0
##### > python-opencv 3.4.1

## Demo
> Download the model https://drive.google.com/file/d/1oDPVqRnWA9hKIN-AWcgr6ViDxrrkhNSX/view?usp=sharing <br/>
> python demo.py to run the demo and visualize the model result in demo_result folder. 

