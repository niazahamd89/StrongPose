## StrongPose: Bottom-up and Strong Keypoint Heat Map Based Pose Estimation.
The adaptation of deep convolutional neural network has made revolutionary advances in human body posture estimation. Towards this, we propose a bottom-up approach to pose estimation and motion recognition. We present StrongPose system that deals with object-part associations using part-based modeling. The convolution network in our model detects strong keypoint heat maps and predicts their comparative displacements, allowing keypoints to be grouped into human instances. Further, it utilizes the keypoints to generate body heat maps that can determine the position of the human body in the image. The StrongPose system is based on fully convolutional engineering and makes proficient inferences while maintaining run-time regardless of the number of individuals in the image. We train and test the StrongPose on the COCO dataset. Evaluation results show that our framework achieves average precision of 0.708 using ResNet-101 and 0.725 using ResNet-152. Our results considerably outperform prior bottom-up frameworks.

- [ ] Dataset: COCO Keypoint 2017 <br/>
- [ ] Platform: TensorFlow, OpenCV <br/>
- [ ] Bacbone Network: ResNet 101, 152 (Modified) <br/>

## Introduction
 Code repo for ICPR 2021 paper (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9413198)

## Visualization Results
img src="pic3.jpeg" width="500" height="600">

## Required
- [ ] Python3
- [ ] Tensorflow 1.80
- [ ] pycocotools 2.0
- [ ] skimage 0.13.0
- [ ] python-opencv 3.4.1

## Demo
- [ ] Download the model https://drive.google.com/file/d/1oDPVqRnWA9hKIN-AWcgr6ViDxrrkhNSX/view?usp=sharing and place in model/personlab folder for quick visualization results. <br/>
- [ ] python demo.py to run the demo and visualize the model result in demo_result folder. 

## Training
- [ ] Download the COCO 2017 dataset 

- [ ] http://images.cocodataset.org/zips/train2017.zip <br/>

- [ ] http://images.cocodataset.org/zips/val2017.zip <br/>

- [ ] http://images.cocodataset.org/annotations/annotations_trainval2017.zip <br/>

- [ ] Place the training images in coco2017/train2017/
- [ ] Place val images in coco2017/val2017/
- [ ] Place training annotations in coco2017/annotations/

- [ ] Download the Resnet101 pretrained model (http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz), put the model in ./model/101/resnet_v2_101.ckpt
- [ ] Edit the config.py to set options for training, e.g. dataset position, input tensor shape, learning rate.
- [ ] Run the train.py script

## Note
- [ ] For any query please comment in issues section. 

