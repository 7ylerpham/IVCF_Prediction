#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, MaxPool2D, Concatenate

import scipy
from skimage import transform
from skimage import io

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import tifffile
get_ipython().system('pip install patchify')


# In[2]:


def plot_all(image):
    index = 0
    count = 1
    fig = plt.figure(figsize = (15, 30))
    for x in range(1, 17):
        for y in range(1, 9):
            plt.subplot(16, 8, count).axis("off")
            plt.title("Count: " + str(count-1))
            plt.imshow(image[:,:,index])
            count += 1
            index += 1
    


# In[3]:


''' Read Images/Masks '''
images = tifffile.imread('ivc_filter_images_84.tif')
masks = tifffile.imread("ivc_filter_masks_84.tif")
print(images.shape)
print(masks.shape)


# In[4]:


''' Splits into training/testing '''
x_train_orig, x_test_orig, y_train_orig, y_test_orig = train_test_split(images, masks, test_size=0.20, random_state=7)
print(x_train_orig.shape)
print(y_train_orig.shape)
print(x_test_orig.shape)
print(y_test_orig.shape)


# In[5]:


''' Creating Patches '''
import patchify
from skimage.util.shape import view_as_blocks, view_as_windows
from patchify import patchify, unpatchify

def to_patches(images, masks):
    image_patches = []
    mask_patches = []
    for img in images:
        img = img/255
        image_patches.append(view_as_blocks(img, (64, 64, 32)))
    for msk in masks:
        # Normalize to 0 or 1
        if msk.max() != 0:
            msk = msk / msk.max()
        mask_patches.append(view_as_blocks(msk, (64, 64, 32)))
    image_patches = np.reshape(image_patches, (-1, 64, 64, 32))
    mask_patches = np.reshape(mask_patches, (-1, 64, 64, 32))
    return image_patches, mask_patches
        
def patches_to_img(patches):
    temp = np.reshape(patches, (4, 4, 4, 64, 64, 32))
    end = unpatchify(temp, (256, 256, 128))
    return end


# In[6]:


''' Split into Patches '''
x1_patches, y1_patches = to_patches(x_train_orig, y_train_orig)
x2_patches, y2_patches = to_patches(x_test_orig, y_test_orig)
print(np.asarray(x1_patches).shape)
print(np.asarray(y1_patches).shape)


# In[7]:


''' Separate the IVCF Patches from non-IVCF Patches'''

x1_zeros = [] # holds the background patches in x_train
x1_ones = [] # holds the ivcf patches in x_train

y1_zeros = [] # holds the background patches in y_train
y1_ones = [] # holds the ivcf patches in y_train

x2_zeros = [] # Same as above but for testing set
x2_ones = []

y2_zeros = []
y2_ones = []

for index in range(np.asarray(x1_patches).shape[0]):
    if y1_patches[index].max() == 1:
        x1_ones.append(x1_patches[index])
        y1_ones.append(y1_patches[index])
    else:
        x1_zeros.append(x1_patches[index])
        y1_zeros.append(y1_patches[index])
        
for index in range(np.asarray(x2_patches).shape[0]):
    if y2_patches[index].max() == 1:
        x2_ones.append(x2_patches[index])
        y2_ones.append(y2_patches[index])
    else:
        x2_zeros.append(x2_patches[index])
        y2_zeros.append(y2_patches[index])
        
print(np.asarray(x1_zeros).shape)
print(np.asarray(x1_ones).shape)
print(np.asarray(y1_zeros).shape)
print(np.asarray(y1_ones).shape)


# In[8]:


print(np.asarray(x2_zeros).shape)
print(np.asarray(x2_ones).shape)
print(np.asarray(y2_zeros).shape)
print(np.asarray(y2_ones).shape)


# In[9]:


''' Create Training/Testing Set with 1:1 Ratio'''
max_len_train = np.asarray(x1_ones).shape[0]
max_len_test = np.asarray(x2_ones).shape[0]

x_train = x1_ones
y_train = y1_ones

# Thresh value for modulus
thresh1 = int(np.asarray(x1_zeros).shape[0] / np.asarray(x1_ones).shape[0])
thresh2 = int(np.asarray(x2_zeros).shape[0] / np.asarray(x2_ones).shape[0])
print(thresh1, thresh2)

for i, val in enumerate(x1_zeros):
    if i % thresh1 == 0 and np.asarray(x_train).shape[0] < max_len_train*2:
        x_train.append(x1_zeros[i])
        y_train.append(y1_zeros[i])


x_test = x2_ones
y_test = y2_ones

for i, val in enumerate(x2_zeros):
    if i % thresh2 == 0 and np.asarray(x_test).shape[0] < max_len_test*2:
        x_test.append(x2_zeros[i])
        y_test.append(y2_zeros[i])


# In[10]:


''' Expand_dims and One Hot Encoding'''
x_train = np.expand_dims(np.asarray(x_train), axis = -1)
x_test = np.expand_dims(np.asarray(x_test), axis = -1)
y_train = to_categorical(np.asarray(y_train))
y_test = to_categorical(np.asarray(y_test))

print(np.asarray(x_train).shape)
print(np.asarray(y_train).shape)
print(np.asarray(x_test).shape)
print(np.asarray(y_test).shape)


# In[11]:


#Define parameters for our model.
channels=1

LR = 0.0001
optim = keras.optimizers.Adam(LR)


# In[12]:


def conv_block(input, num_filters):
    x = Conv3D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)   #Not in the original network. 
    x = Activation("relu")(x)

    x = Conv3D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)  #Not in the original network
    x = Activation("relu")(x)

    return x

#Encoder block: Conv block followed by maxpooling
def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPooling3D((2, 2, 2))(x)
    return x, p   

#Decoder block
#skip features gets input from encoder for concatenation
def decoder_block(input, skip_features, num_filters):
    x = Conv3DTranspose(num_filters, (2, 2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

#Build Unet using the blocks
def build_unet(input_shape, n_classes):
    inputs = Input(input_shape)
    
    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    # s3, p3 = encoder_block(p2, 128)
    # s4, p4 = encoder_block(p3, 256)

    b1 = conv_block(p2, 128) #Bridge

    # d1 = decoder_block(b1, s4, 256)
    # d2 = decoder_block(b1, s3, 128)
    d3 = decoder_block(b1, s2, 64)
    d4 = decoder_block(d3, s1, 32)

    if n_classes == 1:  #Binary
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    outputs = Conv3D(n_classes, 1, padding="same", activation=activation)(d4)  #Change the activation based on n_classes
    print(activation)

    model = Model(inputs, outputs, name="U-Net")
    return model


# In[13]:


METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]


# In[14]:


model = build_unet((64, 64, 32, 1), n_classes=2)
model.compile(optimizer = optim, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=METRICS)
print(model.summary())


# In[15]:


'''Checks'''
print("Input shape", model.input_shape)
print("Output shape", model.output_shape)
print("-------------------")


# In[16]:


history=model.fit(x_train, y_train,
        validation_data=(x_test, y_test),
        batch_size=1,
        epochs=100,
        shuffle=True,
        verbose=1)     


# In[17]:


# Save model for future use
model.save('3D_UNet_patch_normalized.h5')


# In[18]:


def plot_metrics(history):
    metrics = ['loss', 'prc', 'precision', 'recall']
    fig = plt.figure(figsize = (8, 8))
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color='blue', label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color='blue', linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

        plt.legend();


# In[19]:


plot_metrics(history)


# In[6]:


#Load the pretrained model for testing and predictions. 
from tensorflow.keras.models import load_model
my_model = load_model('3D_UNet_patch_normalized.h5', compile=False)


# In[7]:


''' Predict on Training Set '''
img = x_train_orig.copy()
ground_truth = y_train_orig.copy()

# convert training images to patches
img, ground_truth = to_patches(img, ground_truth)
# predict
train_predictions_baseline = my_model.predict(img)
# argmax
train_prediction = np.argmax(train_predictions_baseline, axis=4)[:,:,:,:]

''' Stitch the patches back into Images '''
end = []
gt = []
for x in range(67):
    temp_img = train_prediction[(64*x):64*(x+1)]
    temp_img = patches_to_img(temp_img)
    end.append(temp_img)
for x in range(67):
    temp_img = ground_truth[(64*x):64*(x+1)]
    temp_img = patches_to_img(temp_img)
    gt.append(temp_img)
    
train_predictions = np.asarray(end)
print(train_predictions.shape)


# In[8]:


''' Mean IoU on Training Set '''
from tensorflow.keras.metrics import MeanIoU

n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes) 

gt1 = gt
IOU_keras.update_state(gt1, train_predictions)
print("Training Set:", IOU_keras.result().numpy())


# In[9]:


''' Training Set Per Pixel Basis '''
from sklearn.metrics import confusion_matrix
import seaborn as sns
y_train_matrix = confusion_matrix(np.asarray(gt1).flatten(), np.asarray(train_predictions).flatten())

ax= plt.subplot()
sns.heatmap(y_train_matrix, annot=True, fmt='d', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels'); 
ax.set_title('Train Set Per Pixel Basis'); 
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);


# In[10]:


''' Predict on Testing Set '''
img = x_test_orig.copy()
ground_truth = y_test_orig.copy()

# split images to patches
img, ground_truth = to_patches(img, ground_truth)
# predict
test_predictions_baseline = my_model.predict(img)
# argmax
test_prediction = np.argmax(test_predictions_baseline, axis=4)[:,:,:,:]

''' Stitch patches back into Images '''
end = []
gt = []
for x in range(17):
    temp_img = test_prediction[(64*x):64*(x+1)]
    temp_img = patches_to_img(temp_img)
    end.append(temp_img)
for x in range(17):
    temp_img = ground_truth[(64*x):64*(x+1)]
    temp_img = patches_to_img(temp_img)
    gt.append(temp_img)
    
test_predictions = np.asarray(end)
print(test_predictions.shape)


# In[11]:


''' Mean IoU for Testing Set '''
from tensorflow.keras.metrics import MeanIoU

n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)  
gt2 = gt
IOU_keras.update_state(gt2, test_predictions)
print("Testing Set:", IOU_keras.result().numpy())


# In[12]:


''' Test Set Per Pixel Basis '''
y_test_matrix = confusion_matrix(np.asarray(gt2).flatten(), np.asarray(test_predictions).flatten())

ax= plt.subplot()
sns.heatmap(y_test_matrix, annot=True, fmt='d', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels'); 
ax.set_title('Test Set Per Pixel Basis'); 
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);


# In[ ]:





# In[13]:


def plot_three(image, mask, predicted):
    index = 0
    count = 1
    fig = plt.figure(figsize = (12, 500))
    for x in range(1, 129):
        for y in range(1, 4):
            plt.subplot(128, 3, count).axis("off")
            if count % 3 == 1:
                plt.title("Image Slice: " + str(index))
                plt.imshow(image[:,:,index], cmap='gray')
            elif count % 3 == 2:
                plt.title("Mask Slice: " + str(index))
                plt.imshow(mask[:,:,index])
            else:
                plt.title("Predicted Mask Slice: " + str(index))
                plt.imshow(predicted[:,:,index])
                index += 1
            count += 1


# In[14]:


plot_three(x_test_orig[3], y_test_orig[3], test_predictions[3])


# In[ ]:




