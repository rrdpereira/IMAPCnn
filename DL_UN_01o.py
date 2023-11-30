###############################################################################################################
# DL_UN_01o.py

# Created by: Robson Rogério Dutra Pereira on 12.Sep.2023
# Last Modified: rrdpereira

# Description: Resize or Crop original dataset images and train UNet model.
             # Inference using high resolution and unseen images.

# E-mail: robsondutra.pereira@outlook.com
###############################################################################################################
import sys, time, os, datetime, glob, re
import cv2
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
print(tf.version.VERSION)

import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, \
     Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import generic_utils
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import random
import pandas as pd
from PIL import Image

from platform import python_version
print(f"(Sys version) :|: {sys.version} :|:")
os.system("which python")
print(f"(Python version) :#: {python_version()} :#:")
print("--------------------------->>> os.path.dirname(sys.executable):#: {0}".format(os.path.dirname(sys.executable)))
print("--------------------------->>> re.__version__:#: {0}".format(re.__version__))
print("--------------------------->>> sys.executable:#: {0}".format(sys.executable))
###############################################################################################################
# Allow GPU memory growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# Check for available GPUs
gpu_available = 0 # 0: initial value; 1: GPU available; 2: GPU not available
gpus = tf.config.experimental.list_physical_devices('GPU')
print("--------------------------->>> gpus = {0}".format(gpus)) 
print("--------------------------->>> len(gpus) = {0}".format(len(gpus))) 
if len(gpus) == 1:
    try:
        print("--------------------------->>> gpu_available00 = {0}".format(gpu_available))
        # Use GPU 0 (you can change the index based on your setup)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        gpu_available = 1
        print("--------------------------->>> gpu_available01 = {0}".format(gpu_available))
    except RuntimeError as e:
        print(e)
elif len(gpus) == 0:
    print("--------------------------->>> NOT gpu_available00 = {0}".format(gpu_available)) 
    gpu_available = 2
    print("--------------------------->>> NOT gpu_available01 = {0}".format(gpu_available))

print("--------------------------->>> STATUS gpu_available = {0}".format(gpu_available))

epochsVar = 20
# Loading dataset methods
load_method_ds = 2 # 1: resize; 2: crop
# reading images and resizing them into the same shape
root = "./dataset_aerial_imagery/"
patch_size = 128
patch_size_start = 0
out_folder = "./out/"
os.makedirs(out_folder, exist_ok=True)
out_model = "./saved_models/"
os.makedirs(out_model, exist_ok=True)
out_folder_rgb = "./out/masks_rgb/"
os.makedirs(out_folder_rgb, exist_ok=True)
out_folder_int = "./out/masks_int/"
os.makedirs(out_folder_int, exist_ok=True)
out_folder_hires = "./out/hi_res/"
os.makedirs(out_folder_hires, exist_ok=True)

# Resize images according to patch size
if load_method_ds == 1:
    print("--------------------------->>> 1: resize method")
    imagearray=[]
    if gpu_available == 1:
        print("--------------------------->>> gpu_available = {0}".format(gpu_available))
        with tf.device('/GPU:0'):
            for path,subs,files in os.walk(root):
                dirname=path.split(os.path.sep)[-1]
                if dirname=="images":
                    images=os.listdir(path)
                    images.sort()
                    for i,name in enumerate(images):
                        if name.endswith(".jpg"):
                            image=cv2.imread(path+"/"+name,1)
                            image=cv2.resize(image,(patch_size,patch_size))
                            image=np.array(image)
                            imagearray.append(image)
            print("len(imagearray): {0}".format(len(imagearray)))            
    elif gpu_available == 2:
        print("--------------------------->>> NOT gpu_available = {0}".format(gpu_available))
        for path,subs,files in os.walk(root):
            dirname=path.split(os.path.sep)[-1]
            if dirname=="images":
                images=os.listdir(path)
                images.sort()
                for i,name in enumerate(images):
                    if name.endswith(".jpg"):
                        image=cv2.imread(path+"/"+name,1)
                        image=cv2.resize(image,(patch_size,patch_size))
                        image=np.array(image)
                        imagearray.append(image)
        print("len(imagearray): {0}".format(len(imagearray)))

# Resize masks according to patch size
if load_method_ds == 1:
    print("--------------------------->>> 1: resize method")
    maskarray=[]
    if gpu_available == 1:
        print("--------------------------->>> gpu_available = {0}".format(gpu_available))
        with tf.device('/GPU:0'):
            for path,subs,files in os.walk(root):
                dirname=path.split(os.path.sep)[-1]
                if dirname=="masks":
                    masks=os.listdir(path)
                    masks.sort()
                    for i,M_name in enumerate(masks):
                        if M_name.endswith(".png"):
                            mask =cv2.imread(path+"/"+M_name,1)
                            mask =cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
                            mask =cv2.resize(mask,(patch_size,patch_size))
                            mask =np.array(mask)
                            maskarray.append(mask)
            print("len(maskarray): {0}".format(len(maskarray)))            
    elif gpu_available == 2:
        print("--------------------------->>> NOT gpu_available = {0}".format(gpu_available))    
        for path,subs,files in os.walk(root):
            dirname=path.split(os.path.sep)[-1]
            if dirname=="masks":
                masks=os.listdir(path)
                masks.sort()
                for i,M_name in enumerate(masks):
                    if M_name.endswith(".png"):
                        mask =cv2.imread(path+"/"+M_name,1)
                        mask =cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
                        mask =cv2.resize(mask,(patch_size,patch_size))
                        mask =np.array(mask)
                        maskarray.append(mask)
        print("len(maskarray): {0}".format(len(maskarray)))

# Check if images and masks are OK
if load_method_ds == 1:
    print("--------------------------->>> 1: resize method")
    # image_number = random.randint(0, len(imagearray))
    image_number = 0
    plt.figure(1, figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(imagearray[image_number])
    plt.subplot(122)
    plt.imshow(maskarray[image_number])
    # plt.show()
    # plt.show(block=False)
    # plt.pause(5.0)
    plt.close()
    
    img_name = "sample_00_number" + str(image_number) + ".png"
    b, g, r = cv2.split(maskarray[image_number])  
    rgb_image = cv2.merge((r, g, b))  
    cv2.imwrite(out_folder_rgb + img_name, rgb_image) 
    channel_index = 2  
    single_channel = rgb_image[:, :, 2]
    cv2.imwrite(out_folder_int + img_name, single_channel) 
    
    print("imagearray[image_number].shape: {0}; maskarray[image_number].shape: {1}".format(imagearray[image_number].shape, maskarray[image_number].shape))
    # print("Unique values of maskarray: {0}".format(np.unique(maskarray[image_number])))

if load_method_ds == 1:
    print("--------------------------->>> 1: resize method")
    imagedata = np.array(imagearray)
    maskdata =  np.array(maskarray)

# Check if images and masks are OK
if load_method_ds == 1:
    print("--------------------------->>> 1: resize method")
    # image_number = random.randint(0, len(imagedata))
    image_number = 0
    plt.figure(2, figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(imagedata[image_number])
    plt.subplot(122)
    plt.imshow(maskdata[image_number])
    # plt.show()
    # plt.show(block=False)
    # plt.pause(5.0)
    plt.close()
    
    img_name = "sample_01_number" + str(image_number) + ".png"
    b, g, r = cv2.split(maskdata[image_number])  
    rgb_image = cv2.merge((r, g, b))  
    cv2.imwrite(out_folder_rgb + img_name, rgb_image) 
    channel_index = 2  
    single_channel = rgb_image[:, :, 2]
    cv2.imwrite(out_folder_int + img_name, single_channel) 
    
    print("imagedata[image_number].shape: {0}; maskdata[image_number].shape: {1}".format(imagedata[image_number].shape, maskdata[image_number].shape))
    # print("Unique values of maskdata: {0}".format(np.unique(maskdata[image_number])))

# Crop images according to patch size
if load_method_ds == 2:
    print("--------------------------->>> 2: crop method")
    imagearray=[]
    if gpu_available == 1:
        print("--------------------------->>> gpu_available = {0}".format(gpu_available))
        with tf.device('/GPU:0'):
            for tiles in os.listdir(root):
                if tiles!="classes.json":
                    images_list = os.listdir(os.path.join(root, tiles, "images"))
                    images_list.sort()
                    for img in images_list:
                        img_arr = cv2.imread(os.path.join(root, tiles, "images", img), 1)
                        x=img_arr.shape[0]
                        y=img_arr.shape[1]
                        img_arr = np.moveaxis(img_arr, -1, 0)
                        r1=patch_size_start
                        c1=patch_size
                        for i in range((x//patch_size)):
                            r2=patch_size_start
                            c2=patch_size
                            for j in range((y//patch_size)):
                                l=[]
                                l.append(img_arr[0][r1:c1, r2:c2])
                                l.append(img_arr[1][r1:c1, r2:c2])
                                l.append(img_arr[2][r1:c1, r2:c2])
                                l=np.asarray(l)
                                l=np.moveaxis(l, 0, -1)
                                imagearray.append(l)
                                r2=r2+patch_size
                                c2=c2+patch_size
                            r1=r1+patch_size
                            c1=c1+patch_size                                 
            print(len(imagearray))            
    elif gpu_available == 2:
        print("--------------------------->>> NOT gpu_available = {0}".format(gpu_available))    
        for tiles in os.listdir(root):
            if tiles!="classes.json":
                images_list = os.listdir(os.path.join(root, tiles, "images"))
                images_list.sort()
                for img in images_list:
                    img_arr = cv2.imread(os.path.join(root, tiles, "images", img), 1)
                    x=img_arr.shape[0]
                    y=img_arr.shape[1]
                    img_arr = np.moveaxis(img_arr, -1, 0)
                    r1=patch_size_start
                    c1=patch_size
                    for i in range((x//patch_size)):
                        r2=patch_size_start
                        c2=patch_size
                        for j in range((y//patch_size)):
                            l=[]
                            l.append(img_arr[0][r1:c1, r2:c2])
                            l.append(img_arr[1][r1:c1, r2:c2])
                            l.append(img_arr[2][r1:c1, r2:c2])
                            l=np.asarray(l)
                            l=np.moveaxis(l, 0, -1)
                            imagearray.append(l)
                            r2=r2+patch_size
                            c2=c2+patch_size
                        r1=r1+patch_size
                        c1=c1+patch_size                                 
        print(len(imagearray))

# Crop masks according to patch size
if load_method_ds == 2:
    print("--------------------------->>> 2: crop method")
    maskarray=[]
    if gpu_available == 1:
        print("--------------------------->>> gpu_available = {0}".format(gpu_available))
        with tf.device('/GPU:0'):
            for tiles in os.listdir(root):    
                if tiles!="classes.json":
                    images_list = os.listdir(os.path.join(root, tiles, "masks"))
                    images_list.sort()
                    for img in images_list:            
                        img_arr = cv2.imread(os.path.join(root, tiles, "masks", img), 1)
                        img_arr =cv2.cvtColor(img_arr,cv2.COLOR_BGR2RGB)          
                        x=img_arr.shape[0]
                        y=img_arr.shape[1]
                        img_arr = np.moveaxis(img_arr, -1, 0)
                        r1=patch_size_start
                        c1=patch_size
                        for i in range((x//patch_size)):
                            r2=patch_size_start
                            c2=patch_size
                            for j in range((y//patch_size)):                    
                                l=[]
                                l.append(img_arr[0][r1:c1, r2:c2])
                                l.append(img_arr[1][r1:c1, r2:c2])
                                l.append(img_arr[2][r1:c1, r2:c2])
                                l=np.asarray(l)
                                l=np.moveaxis(l, 0, -1)
                                maskarray.append(l)
                                r2=r2+patch_size
                                c2=c2+patch_size
                            r1=r1+patch_size
                            c1=c1+patch_size
            print(len(maskarray))            
    elif gpu_available == 2:
        print("--------------------------->>> NOT gpu_available = {0}".format(gpu_available))
        for tiles in os.listdir(root):    
            if tiles!="classes.json":
                images_list = os.listdir(os.path.join(root, tiles, "masks"))
                images_list.sort()
                for img in images_list:            
                    img_arr = cv2.imread(os.path.join(root, tiles, "masks", img), 1)
                    img_arr =cv2.cvtColor(img_arr,cv2.COLOR_BGR2RGB)          
                    x=img_arr.shape[0]
                    y=img_arr.shape[1]
                    img_arr = np.moveaxis(img_arr, -1, 0)
                    r1=patch_size_start
                    c1=patch_size
                    for i in range((x//patch_size)):
                        r2=patch_size_start
                        c2=patch_size
                        for j in range((y//patch_size)):                    
                            l=[]
                            l.append(img_arr[0][r1:c1, r2:c2])
                            l.append(img_arr[1][r1:c1, r2:c2])
                            l.append(img_arr[2][r1:c1, r2:c2])
                            l=np.asarray(l)
                            l=np.moveaxis(l, 0, -1)
                            maskarray.append(l)
                            r2=r2+patch_size
                            c2=c2+patch_size
                        r1=r1+patch_size
                        c1=c1+patch_size
        print(len(maskarray))                

# Check if images and masks are OK
if load_method_ds == 2:
    print("--------------------------->>> 2: crop method")
    # image_number = random.randint(0, len(imagearray))
    image_number = 0
    plt.figure(1, figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(imagearray[image_number])
    plt.subplot(122)
    plt.imshow(maskarray[image_number])
    # plt.show()
    # plt.show(block=False)
    # plt.pause(5.0)
    plt.close()

    img_name = "sample_02_number" + str(image_number) + ".png"
    b, g, r = cv2.split(maskarray[image_number])  
    rgb_image = cv2.merge((r, g, b))  
    cv2.imwrite(out_folder_rgb + img_name, rgb_image) 
    channel_index = 2  
    single_channel = rgb_image[:, :, 2]
    cv2.imwrite(out_folder_int + img_name, single_channel) 
    
    print("imagearray[image_number].shape: {0}; maskarray[image_number].shape: {1}".format(imagearray[image_number].shape, maskarray[image_number].shape))
    # print("Unique values of maskarray: {0}".format(np.unique(maskarray[image_number])))

if load_method_ds == 2:
    print("--------------------------->>> 2: crop method")
    imagedata = np.array(imagearray)
    maskdata =  np.array(maskarray)

# Check if images and masks are OK
if load_method_ds == 2:
    print("--------------------------->>> 2: crop method")
    # image_number = random.randint(0, len(imagedata))
    image_number = 0
    plt.figure(2, figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(imagedata[image_number])
    plt.subplot(122)
    plt.imshow(maskdata[image_number])
    # plt.show()
    # plt.show(block=False)
    # plt.pause(5.0)
    plt.close()

    img_name = "sample_03_number" + str(image_number) + ".png"
    b, g, r = cv2.split(maskdata[image_number])  
    rgb_image = cv2.merge((r, g, b))  
    cv2.imwrite(out_folder_rgb + img_name, rgb_image) 
    channel_index = 2  
    single_channel = rgb_image[:, :, 2]
    cv2.imwrite(out_folder_int + img_name, single_channel) 

    print("imagedata[image_number].shape: {0}; maskdata[image_number].shape: {1}".format(imagedata[image_number].shape, maskdata[image_number].shape))
    # print("Unique values of maskdata: {0}".format(np.unique(maskdata[image_number])))

# Check if images and masks are OK
n = 3
plt.figure(3, figsize=(12, 6))
plt.imshow(imagedata[n])
# plt.show()
# plt.show(block=False)
# plt.pause(5.0)
plt.close()
plt.imshow(maskdata[n])
# plt.show()
# plt.show(block=False)
# plt.pause(5.0)
plt.close()

Building=np.array([60,16,152])
Land=np.array([132,41,246])
Road=np.array([110,193,228])
Vegetation=np.array([254,221,58])
Water=np.array([226,169,41])
Unlabeled =np.array([155,155,155])

label = maskdata

# Check if images and masks are OK
# image_number = random.randint(0, len(imagedata))
image_number = 0
plt.figure(4, figsize=(12, 6))
plt.subplot(121)
plt.imshow(imagedata[image_number])
plt.subplot(122)
plt.imshow(label[image_number])
# plt.show()
# plt.show(block=False)
# plt.pause(5.0)
plt.close()

img_name = "sample_04_number" + str(image_number) + ".png"
b, g, r = cv2.split(label[image_number])  
rgb_image = cv2.merge((r, g, b))  
cv2.imwrite(out_folder_rgb + img_name, rgb_image) 
channel_index = 2  
single_channel = rgb_image[:, :, 2]
cv2.imwrite(out_folder_int + img_name, single_channel) 

print("imagedata[image_number].shape: {0}; label[image_number].shape: {1}".format(imagedata[image_number].shape, label[image_number].shape))
# print("Unique values of label: {0}".format(np.unique(label[image_number])))

def flatLabels(label):
    label_seg = np.zeros(label.shape,dtype=np.uint8)
    label_seg [np.all(label==Building,axis=-1)] = 0
    label_seg [np.all(label==Land,axis=-1)] = 1
    label_seg [np.all(label==Road,axis=-1)] = 2
    label_seg [np.all(label==Vegetation,axis=-1)] = 3
    label_seg [np.all(label==Water,axis=-1)] = 4
    label_seg [np.all(label==Unlabeled,axis=-1)] = 5
    
    label_seg = label_seg[:,:,0]
    
    return label_seg

labels = []
if gpu_available == 1:
    print("--------------------------->>> gpu_available = {0}".format(gpu_available))
    flatLabels_00_start = time.time()
    print('START flatLabels in: {0} seconds'.format(flatLabels_00_start))   
    with tf.device('/GPU:0'):  
        for i in range(maskdata.shape[0]):
            label = flatLabels(maskdata[i])
            labels.append(label)
    flatLabels_00_end = time.time() - flatLabels_00_start
    print('END Complete flatLabels in: {0} seconds| {1} minutes'.format(flatLabels_00_end, (flatLabels_00_end/60)))          
elif gpu_available == 2:
    print("--------------------------->>> NOT gpu_available = {0}".format(gpu_available))
    flatLabels_00_start = time.time()
    print('START flatLabels in: {0} seconds'.format(flatLabels_00_start))     
    for i in range(maskdata.shape[0]):
        label = flatLabels(maskdata[i])
        labels.append(label)
    flatLabels_00_end = time.time() - flatLabels_00_start
    print('END Complete flatLabels in: {0} seconds| {1} minutes'.format(flatLabels_00_end, (flatLabels_00_end/60)))         

# Check if images and masks are OK
# image_number = random.randint(0, len(imagedata))
image_number = 0
plt.figure(5, figsize=(12, 6))
plt.subplot(121)
plt.imshow(imagedata[image_number])
plt.subplot(122)
plt.imshow(labels[image_number])
# plt.show()
# plt.show(block=False)
# plt.pause(5.0)
plt.close()

img_name = "sample_05_number" + str(image_number) + ".png"
cv2.imwrite(out_folder_rgb + img_name, labels[image_number]) 
channel_index = 2  
single_channel = rgb_image[:, :, 2]
cv2.imwrite(out_folder_int + img_name, labels[image_number]) 

print("imagedata[image_number].shape: {0}; labels[image_number].shape: {1}".format(imagedata[image_number].shape, labels[image_number].shape))
# print("Unique values of labels: {0}".format(np.unique(labels[image_number])))

labels = np.array(labels)

# Check if images and masks are OK
# image_number = random.randint(0, len(imagedata))
image_number = 0
plt.figure(6, figsize=(12, 6))
plt.subplot(121)
plt.imshow(imagedata[image_number])
plt.subplot(122)
plt.imshow(labels[image_number])
# plt.show()
# plt.show(block=False)
# plt.pause(5.0)
plt.close()

img_name = "sample_06_number" + str(image_number) + ".png"
cv2.imwrite(out_folder_rgb + img_name, labels[image_number]) 
channel_index = 2  
single_channel = rgb_image[:, :, 2]
cv2.imwrite(out_folder_int + img_name, labels[image_number]) 

print("imagedata[image_number].shape: {0}; labels[image_number].shape: {1}".format(imagedata[image_number].shape, labels[image_number].shape))
# print("Unique values of labels: {0}".format(np.unique(labels[image_number])))

labels = np.expand_dims(labels, axis=3)

# Check if images and masks are OK
# image_number = random.randint(0, len(imagedata))
image_number = 0
plt.figure(7, figsize=(12, 6))
plt.subplot(121)
plt.imshow(imagedata[image_number])
plt.subplot(122)
plt.imshow(labels[image_number])
# plt.show()
plt.show(block=False)
plt.pause(5.0)
plt.close()

img_name = "sample_07_number" + str(image_number) + ".png"
cv2.imwrite(out_folder_rgb + img_name, labels[image_number]) 
channel_index = 2  
single_channel = rgb_image[:, :, 2]
cv2.imwrite(out_folder_int + img_name, labels[image_number]) 

print("imagedata[image_number].shape: {0}; labels[image_number].shape: {1}".format(imagedata[image_number].shape, labels[image_number].shape))
# print("Unique values of labels: {0}".format(np.unique(labels[image_number])))

n_classes = len(np.unique(labels))
print("n_classes: {0}".format(n_classes))

if gpu_available == 1:
    print("--------------------------->>> gpu_available = {0}".format(gpu_available))
    one_hot_encoded_00_start = time.time()
    print('START one_hot_encoded in: {0} seconds'.format(one_hot_encoded_00_start))    
    with tf.device('/GPU:0'): 
        labels_cat = to_categorical(labels, num_classes=n_classes)
    one_hot_encoded_00_end = time.time() - one_hot_encoded_00_start
    print('END Complete one_hot_encoded in: {0} seconds| {1} minutes'.format(one_hot_encoded_00_end, (one_hot_encoded_00_end/60)))          
elif gpu_available == 2:
    print("--------------------------->>> NOT gpu_available = {0}".format(gpu_available))
    one_hot_encoded_00_start = time.time()
    print('START one_hot_encoded in: {0} seconds'.format(one_hot_encoded_00_start))     
    labels_cat = to_categorical(labels, num_classes=n_classes)
    one_hot_encoded_00_end = time.time() - one_hot_encoded_00_start
    print('END Complete one_hot_encoded in: {0} seconds| {1} minutes'.format(one_hot_encoded_00_end, (one_hot_encoded_00_end/60)))    

img_height, img_width, img_channels = imagedata[image_number].shape
print("img_height: {0}; img_width: {1}; img_channels: {2}".format(img_height, img_width, img_channels))
# splitting data
X_train, X_test, y_train, y_test = train_test_split(imagedata, labels_cat, test_size = 0.20)
print("len(imagedata): {0}; len(labels_cat): {1}; len(X_train): {2}; len(X_test): {3}; len(y_train): {4}; len(y_test): {5}".format(len(imagedata),len(labels_cat),len(X_train),len(X_test),len(y_train),len(y_test)))
weights = [0.155, 0.155, 0.155, 0.155, 0.155, 0.155]
dice_loss = sm.losses.DiceLoss(class_weights=weights)
focal = sm.losses.CategoricalFocalLoss()
total = dice_loss + (1*focal)

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

print("n_classes: {0}; patch_size: {1}; img_channels: {2}".format(n_classes, patch_size, img_channels))

def unet(n_classes=n_classes, IMG_HEIGHT=patch_size, IMG_WIDTH=patch_size, IMG_CHANNELS=img_channels): 
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs
    # begin with contraction part
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    # begin with expantion part
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    # define output layer
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.summary()

    print("model.input_shape: {0}".format(model.input_shape))
    print("model.output_shape: {0}".format(model.output_shape))

    return model

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

print("n_classes: {0}; patch_size: {1}; img_channels: {2};\nIMG_HEIGHT: {3}; IMG_WIDTH: {4}; IMG_CHANNELS: {5}".format(n_classes, patch_size, img_channels, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

def get_model():
    return unet(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

metrics=['accuracy', jacard_coef]

model = get_model()
model.compile(optimizer='adam', loss=total, metrics=metrics)

# Modelcheckpoint
file_name = "Sematic_Segmentation_Model_UNet" + time.strftime("_%Y%m%d_%H%M%S")
checkpoint_path = "./logs/" + file_name + "/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', mode='min'),
             tf.keras.callbacks.TensorBoard(log_dir="./logs/{}".format(file_name))]

history1 = model.fit(X_train, y_train,
                    batch_size = 16,
                    verbose=1,
                    epochs=epochsVar,
                    validation_data=(X_test, y_test),
                    callbacks=callbacks,
                    shuffle=False)

history = history1
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure(8, figsize=(6, 6))
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training + validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(out_folder + "sample_08_val_loss_Infc.png")
# plt.show()
plt.show(block=False)
plt.pause(5.0)
plt.close()

acc = history.history['jacard_coef']
val_acc = history.history['val_jacard_coef']

plt.figure(9, figsize=(6, 6))
plt.plot(epochs, acc, 'y', label='Training IoU')
plt.plot(epochs, val_acc, 'r', label='Validation IoU')
plt.title('Training and validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.savefig(out_folder + "sample_09_val_jacard_coef_Infc.png")
# plt.show()
plt.show(block=False)
plt.pause(5.0)
plt.close()

y_pred=model.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)
y_test_argmax=np.argmax(y_test, axis=3)

test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test_argmax[test_img_number]
test_img_input=np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]

plt.figure(10, figsize=(12, 4))
plt.subplot(131)
plt.title('Testing Image')
plt.imshow(test_img)
plt.subplot(132)
plt.title('Testing Label')
plt.imshow(ground_truth)
plt.subplot(133)
plt.title('Prediction on test image')
plt.imshow(predicted_img)
# plt.show()
plt.show(block=False)
plt.pause(5.0)
plt.close()

img_name = "sample_10_inference_groundtruth.png"
cv2.imwrite(out_folder_rgb + img_name, ground_truth) 
cv2.imwrite(out_folder_int + img_name, ground_truth) 
img_name2 = "sample_11_inference_predicted.png"
cv2.imwrite(out_folder_rgb + img_name2, predicted_img) 
cv2.imwrite(out_folder_int + img_name2, predicted_img) 

if gpu_available == 1:
    print("--------------------------->>> gpu_available = {0}".format(gpu_available))  
    def crop_image(rgb_img):
        with tf.device('/GPU:0'):
            x=rgb_img.shape[0]
            y=rgb_img.shape[1]
            plt.figure(14, figsize=(6, 4))
            rgb_imgL = rgb_img
            plt.imshow(rgb_imgL)
            # plt.show()
            plt.show(block=False)
            plt.pause(5.0)
            plt.close()
            rgb_img = np.moveaxis(rgb_img, -1, 0)
            predicted_arr = np.full(((x//patch_size)*patch_size, (y//patch_size)*patch_size), 0)
            r1=patch_size_start
            c1=patch_size
            for i in range((x//patch_size)):
                r2=patch_size_start
                c2=patch_size
                for j in range((y//patch_size)):
                    l=[]
                    l.append(rgb_img[0][r1:c1, r2:c2])
                    l.append(rgb_img[1][r1:c1, r2:c2])
                    l.append(rgb_img[2][r1:c1, r2:c2])
                    l=np.asarray(l)
                    l=np.moveaxis(l, 0, -1)
                    test_img_input=np.expand_dims(l, 0)
                    prediction = (model.predict(test_img_input))
                    predicted_img=np.argmax(prediction, axis=3)[0,:,:]
                    predicted_arr[r1:c1, r2:c2] = predicted_img
                    r2=r2+patch_size
                    c2=c2+patch_size
                r1=r1+patch_size
                c1=c1+patch_size
            unique_values = sorted(set(predicted_arr.flatten()))
            print(unique_values)
            class_names = []
            if len(unique_values) == 3:
                print(unique_values)
                class_names = [
                    "Class_" + str(unique_values[0]),
                    "Class_" + str(unique_values[1]),
                    "Class_" + str(unique_values[2]),
                ]
            elif len(unique_values) == 4:
                print(unique_values)
                class_names = [
                    "Class_" + str(unique_values[0]),
                    "Class_" + str(unique_values[1]),
                    "Class_" + str(unique_values[2]),
                    "Class_" + str(unique_values[3]),
                ]  
            elif len(unique_values) == 5:
                print(unique_values)
                class_names = [
                    "Bui_" + str(unique_values[0]),
                    "Lad_" + str(unique_values[1]),
                    "Rod_" + str(unique_values[2]),
                    "Veg_" + str(unique_values[3]),
                    "Wat_" + str(unique_values[4]),
                ]
            elif len(unique_values) == 6:
                print(unique_values)
                class_names = [
                    "Bui_" + str(unique_values[0]),
                    "Lad_" + str(unique_values[1]),
                    "Rod_" + str(unique_values[2]),
                    "Veg_" + str(unique_values[3]),
                    "Wat_" + str(unique_values[4]),
                    "Unl_" + str(unique_values[5]),
                ]
                        
            plt.figure(15)
            fig, axs = plt.subplots(1, 3, figsize=(18, 4))
            axs[0].imshow(rgb_imgL)
            axs[0].set_title('Original Image')
            axs[1].imshow(predicted_arr, cmap='gray')
            axs[1].set_title('Grayscale Image')
            legend_height = 50
            legend_width = len(unique_values) * 50
            legend = np.zeros((legend_height, legend_width), dtype=np.uint8)
            color_index = 0
            for value in unique_values:
                cv2.rectangle(
                    legend,
                    (color_index * 50, 0),
                    ((color_index + 1) * 50, legend_height),
                    int(value),
                    thickness=cv2.FILLED,
                )
                color_index += 1
            axs[2].imshow(legend, cmap='gray')
            axs[2].set_title('Legend')
            for ax in axs:
                ax.axis('off')
            legend_center_x = legend_width // 2
            class_name_x = np.linspace(0, legend_width, len(class_names))
            class_name_x -= (class_name_x[-1] / 2)
            class_name_y = legend_height + 10
            for icl, class_name in enumerate(class_names):
                axs[2].text(class_name_x[icl] + legend_center_x, class_name_y, class_name, ha='center')
            plt.savefig(out_folder_hires + "predicted_arr_Infc" + str(i) + ".png")                   
            # plt.show()
            plt.show(block=False)
            plt.pause(5.0)
            plt.close()

            plt.figure(17,figsize=(6, 6))
            plt.imshow(predicted_arr)
            # plt.show()
            plt.show(block=False)
            plt.pause(5.0)
            plt.close()
        
            img_name2 = "sample_12_inference_predicted" + str(i) + ".png"
            cv2.imwrite(out_folder_hires + img_name2, predicted_arr) 
          
elif gpu_available == 2:
    print("--------------------------->>> NOT gpu_available = {0}".format(gpu_available))
    def crop_image(rgb_img):
        x=rgb_img.shape[0]
        y=rgb_img.shape[1]
        plt.figure(14, figsize=(6, 4))
        rgb_imgL = rgb_img
        plt.imshow(rgb_imgL)
        # plt.show()
        plt.show(block=False)
        plt.pause(5.0)
        plt.close()
        rgb_img = np.moveaxis(rgb_img, -1, 0)
        predicted_arr = np.full(((x//patch_size)*patch_size, (y//patch_size)*patch_size), 0)
        r1=patch_size_start
        c1=patch_size
        for i in range((x//patch_size)):
            r2=patch_size_start
            c2=patch_size
            for j in range((y//patch_size)):
                l=[]
                l.append(rgb_img[0][r1:c1, r2:c2])
                l.append(rgb_img[1][r1:c1, r2:c2])
                l.append(rgb_img[2][r1:c1, r2:c2])
                l=np.asarray(l)
                l=np.moveaxis(l, 0, -1)
                test_img_input=np.expand_dims(l, 0)
                prediction = (model.predict(test_img_input))
                predicted_img=np.argmax(prediction, axis=3)[0,:,:]
                predicted_arr[r1:c1, r2:c2] = predicted_img
                r2=r2+patch_size
                c2=c2+patch_size
            r1=r1+patch_size
            c1=c1+patch_size
        unique_values = sorted(set(predicted_arr.flatten()))
        print(unique_values)
        class_names = []
        if len(unique_values) == 3:
            print(unique_values)
            class_names = [
                "Class_" + str(unique_values[0]),
                "Class_" + str(unique_values[1]),
                "Class_" + str(unique_values[2]),
            ]
        elif len(unique_values) == 4:
            print(unique_values)
            class_names = [
                "Class_" + str(unique_values[0]),
                "Class_" + str(unique_values[1]),
                "Class_" + str(unique_values[2]),
                "Class_" + str(unique_values[3]),
            ]  
        elif len(unique_values) == 5:
            print(unique_values)
            class_names = [
                "Bui_" + str(unique_values[0]),
                "Lad_" + str(unique_values[1]),
                "Rod_" + str(unique_values[2]),
                "Veg_" + str(unique_values[3]),
                "Wat_" + str(unique_values[4]),
            ]
        elif len(unique_values) == 6:
            print(unique_values)
            class_names = [
                "Bui_" + str(unique_values[0]),
                "Lad_" + str(unique_values[1]),
                "Rod_" + str(unique_values[2]),
                "Veg_" + str(unique_values[3]),
                "Wat_" + str(unique_values[4]),
                "Unl_" + str(unique_values[5]),
            ]
               
        plt.figure(15)
        fig, axs = plt.subplots(1, 3, figsize=(18, 4))
        axs[0].imshow(rgb_imgL)
        axs[0].set_title('Original Image')
        axs[1].imshow(predicted_arr, cmap='gray')
        axs[1].set_title('Grayscale Image')
        legend_height = 50
        legend_width = len(unique_values) * 50
        legend = np.zeros((legend_height, legend_width), dtype=np.uint8)
        color_index = 0
        for value in unique_values:
            cv2.rectangle(
                legend,
                (color_index * 50, 0),
                ((color_index + 1) * 50, legend_height),
                int(value),
                thickness=cv2.FILLED,
            )
            color_index += 1
        axs[2].imshow(legend, cmap='gray')
        axs[2].set_title('Legend')
        for ax in axs:
            ax.axis('off')
        legend_center_x = legend_width // 2
        class_name_x = np.linspace(0, legend_width, len(class_names))
        class_name_x -= (class_name_x[-1] / 2)
        class_name_y = legend_height + 10
        for icl, class_name in enumerate(class_names):
            axs[2].text(class_name_x[icl] + legend_center_x, class_name_y, class_name, ha='center')
        plt.savefig(out_folder_hires + "predicted_arr_Infc" + str(i) + ".png")                   
        # plt.show()
        plt.show(block=False)
        plt.pause(5.0)
        plt.close()

        plt.figure(17,figsize=(6, 6))
        plt.imshow(predicted_arr)
        # plt.show()
        plt.show(block=False)
        plt.pause(5.0)
        plt.close()
    
        img_name2 = "sample_12_inference_predicted" + str(i) + ".png"
        cv2.imwrite(out_folder_hires + img_name2, predicted_arr) 

image_path = [
              "./sample/image_part_003.jpg",
              "./sample/image_part_007.jpg",
              "./sample/image_part_008.jpg",
              ]

for i in image_path:
    rgb_img_00_start = time.time()
    print('START rgb_img Inference in: {0} seconds'.format(rgb_img_00_start)) 
    img = cv2.imread(i, cv2.IMREAD_COLOR)
    crop_image(img)
    rgb_img_encoded_00_end = time.time() - rgb_img_00_start
    print('END Complete rgb_img Inference in: {0} seconds| {1} minutes'.format(rgb_img_encoded_00_end, (rgb_img_encoded_00_end/60)))      

# Model
model_filename = out_model + file_name + ".h5"

if os.path.exists(model_filename):
    os.remove(model_filename)

model.save(model_filename)