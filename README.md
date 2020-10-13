## StrongPose: Bottom-up, Strong Keypoints Heat Maps Base Person Pose Estimation.
This work established a StrongPose model which jointly tackle the problem of pose estimation and person detection. Towards this purpose we present a bottom-up box-free approach for the task of pose estimation and action recognition. The model utilizes a convolution network that learns how to detect Strong Keypoints Heat Maps (SKHM) and predict their comparative displacements, enabling us to group keypoints into person pose instances. Further, we produce Body Heat Maps (BHM) with the help of keypoints which allows us to localize human body in the picture. The StrongPose framework is based on a fully-convolutional engineering and permits proficient inference, with runtime basically autonomous of the number of individuals display within the scene. Train and test on COCO data alone, our framework achieves COCO test-dev keypoint average precision of 0.708 using ResNet-101 and 0.725 using ResNet-152, which considerably outperforms all prior bottom-up pose estimation frameworks.

###Dataset: COCO Keypoint 2017
###Platform: TensorFlow, OpenCV
###Bacboon Network: ResNet 101, 152 (Modified)
