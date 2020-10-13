import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf 
from config import config
import model
from data_generator import DataGeneraotr
import numpy as np
from skimage import io
from plot import *
from post_proc import *
import cv2
from post_proc import get_keypoints

multiscale = [1,1,1]
save_path = './demo_result/'

# build the model
batch_size,height,width=1,config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1]
tf_img = []
outputs = []
for i in range(len(multiscale)):
    scale = multiscale[i]
    tf_img.append(tf.placeholder(tf.float32,shape=[batch_size,int(scale*height),int(scale*width),3]))
    outputs.append(model.model(tf_img[i])) 
sess = tf.Session()

# load the parameters
global_vars = tf.global_variables()
saver = tf.train.Saver(var_list = global_vars)
checkpoint_path = './model/StrongPose/'+'model.ckpt-11'
saver.restore(sess,checkpoint_path)
print("Trained Model Restored!")

# input the demo image
dataset = DataGeneraotr()

scale_outputs = []
for i in range(len(multiscale)):
    scale = multiscale[i]
    scale_img = dataset.get_multi_scale_img(give_id=11615,scale=scale)
    if i==0:
        img = scale_img[:,:,[2,1,0]]
        plt.imsave(save_path+'input_image.jpg',img)
    imgs_batch = np.zeros((batch_size,int(scale*height),int(scale*width),3))
    imgs_batch[0] = scale_img

    # make prediction  
    one_scale_output = sess.run(outputs[i],feed_dict={tf_img[i]:imgs_batch})
    scale_outputs.append([o[0] for o in one_scale_output])

sample_output = scale_outputs[0]
for i in range(1,len(multiscale)):
    for j in range(len(sample_output)):
        sample_output[j]+=scale_outputs[i][j]
for j in range(len(sample_output)):
    sample_output[j] /=len(multiscale)

# visualization
print('Visualization image has been saved into '+save_path)

def overlay(img, over, alpha=0.5):
    out = img.copy()
    if img.max() > 2.:
        out = out / 400.
    out *= 1-alpha
    if len(over.shape)==2:
        out += alpha*over[:,:,np.newaxis]
    else:
        out += alpha*over    
    return out

# Here is the output map for all keypoint

#head = sample_output [0][:,:,0]+ sample_output[0][:,:,1]+sample_output [0][:,:,3]+ sample_output[0][:,:,5]+sample_output [0][:,:,6]+ sample_output[0][:,:,8]+ sample_output [0][:,:,10]+ sample_output[0][:,:,12]

#head = sample_output [0][:,:,3]
#Rwrist = sample_output [0][:,:,13]
#Lelbow = sample_output [0][:,:,5]
#Rwrist = sample_output [0][:,:,16]
#n = sample_output [0] [:,:, 12]
#m = sample_output [0] [:,:, 15]

Rshoulder_map = sample_output[0][:,:,config.KEYPOINTS.index('Rshoulder')]
#print("Right Shoulder")
Lknee = sample_output[0][:,:,config.KEYPOINTS.index('Lknee')]
#print("Left Knee")
nose = sample_output[0][:,:,config.KEYPOINTS.index('nose')]
#Lshoulder = sample_output[0][:,:,config.KEYPOINTS.index('Lshoulder')]
Relbow = sample_output[0][:,:,config.KEYPOINTS.index('Relbow')]
Rwrist = sample_output[0][:,:,config.KEYPOINTS.index('Rwrist')]
Lelbow = sample_output[0][:,:,config.KEYPOINTS.index('Lelbow')]
Lwrist = sample_output[0][:,:,config.KEYPOINTS.index('Lwrist')]
Rhip = sample_output[0][:,:,config.KEYPOINTS.index('Rhip')]
Lhip = sample_output[0][:,:,config.KEYPOINTS.index('Lhip')]

Rknee = sample_output[0][:,:,config.KEYPOINTS.index('Rknee')]
Lankle = sample_output[0][:,:,config.KEYPOINTS.index('Lankle')]
Rankle = sample_output[0][:,:,config.KEYPOINTS.index('Rankle')] 



result = Rshoulder_map + Lknee + nose + Relbow + Rwrist + Lelbow + Lwrist + Rhip + Rknee + Lankle + Lhip + Rankle

plt.imsave(save_path+'kp_map.jpg',overlay(img, result, alpha=0.4))


# Gaussian filtering helps when there are multiple local maxima for the same keypoint.
H = compute_heatmaps(kp_maps=sample_output[0], short_offsets=sample_output[1])
for i in range(17):
    H[:,:,i] = gaussian_filter(H[:,:,i], sigma=1)
    #H[:, :, i] = peak_local_max(H[:, :, i], min_distance=1)
    
#plt.imsave(save_path+'heatmaps.jpg',H[:,:,config.KEYPOINTS.index('Rshoulder')]*10)
    
heat_img = plt.imsave(save_path+'heatmaps.jpg',H[:,:,0]+H[:,:,1]+H[:,:,2]+H[:,:,3]+H[:,:,4]+H[:,:,5]+H[:,:,6]+H[:,:,7]+H[:,:,8]+H[:,:,9]+H[:,:,10]+H[:,:,11]+H[:,:,12]+H[:,:,13]+H[:,:,14]+H[:,:,15]+H[:,:,16])

# The heatmaps are computed using the short offsets predicted by the network
# Here are the right shoulder offsets
visualize_short_offsets(offsets=sample_output[1], heatmaps=H, keypoint_id='Rshoulder', img=img, every=8,save_path=save_path)

# The connections between keypoints are computed via the mid-range offsets.
# We can visuzalize them as well; for example right shoulder -> right hip
visualize_mid_offsets(offsets= sample_output[2], heatmaps=H, from_kp='Rshoulder', to_kp='Rhip', img=img, every=8,save_path=save_path)

# And we can see the reverse connection (Rhip -> Rshjoulder) as well
# visualize_mid_offsets(offsets= sample_output[2], heatmaps=H, to_kp='Rshoulder', from_kp='Rhip', img=img, every=8,save_path=save_path)

# We can use the heatmaps to compute the skeletons
pred_kp = get_keypoints(H)
print(pred_kp)
pred_skels = group_skeletons(keypoints=pred_kp, mid_offsets=sample_output[2])
pred_skels = [skel for skel in pred_skels if (skel[:,2]>0).sum() > 4]
print ('Number of detected skeletons: {}'.format(len(pred_skels)))

plot_poses(img, pred_skels,save_path=save_path)

# we can use the predicted skeletons along with the long-range offsets and binary segmentation mask to compute the instance masks. 
plt.imsave(save_path+'segmentation_mask.jpg',apply_mask(img, sample_output[4][:,:,0]>0.5, color=[255,255,200]))

visualize_long_offsets(offsets=sample_output[3], keypoint_id='Rshoulder', seg_mask=sample_output[4], img=img, every=8,save_path=save_path)

instance_masks = get_instance_masks(pred_skels, sample_output[-1][:,:,0], sample_output[-2])
plot_instance_masks(instance_masks, img,save_path=save_path)