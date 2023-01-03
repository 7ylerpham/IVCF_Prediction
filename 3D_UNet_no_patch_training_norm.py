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
from skimage.transform import resize

import tifffile


# In[2]:


def plot_all(image):
    index = 0
    count = 1
    fig = plt.figure(figsize = (15, 30))
    for x in range(1, 17):
        for y in range(1, 9):
            plt.subplot(16, 8, count).axis("off")
            plt.title("Count: " + str(count-1))
            plt.imshow(image[:,:,index], cmap='gray')
            count += 1
            index += 1
    


# In[3]:


CROP_RATE = 0.2

def cropScan2(scan):  
    col_size = scan.shape[0]
    row_size = scan.shape[1]

    
    newScan = scan[int(CROP_RATE*col_size):int((1-CROP_RATE)*col_size),
                   int(CROP_RATE*row_size):int((1-CROP_RATE)*row_size),:]
    return newScan


# In[4]:


''' Read Images/Masks '''
images = tifffile.imread('ivc_filter_images_84.tif')
masks = tifffile.imread("ivc_filter_masks_84.tif")
print(images.shape)
print(masks.shape)


# In[5]:


''' Crop 20% on all sides'''
final_images = []
for img in images:
    final_images.append(cropScan2(img/255))
    
final_masks = []
for mask in masks:
    mask = mask/mask.max()
    final_masks.append(cropScan2(mask))
    
final_images = np.asarray(final_images)
final_masks = np.asarray(final_masks)
print(final_images.shape, final_masks.shape)
print(np.unique(final_masks))


# In[6]:


''' Resize to (128, 128, 128) '''
final_images = resize(final_images, (84, 128, 128, 128))
final_masks = resize(final_masks, (84, 128, 128, 128))
print(final_images.shape, final_masks.shape)
print(np.unique(final_masks))


# In[7]:


fig = plt.figure(figsize = (12, 12))
plt.subplot(2, 2, 1).axis("off")
plt.title("Original Image")
plt.imshow(images[0][:,:,54], cmap='gray')
          
plt.subplot(2, 2, 2).axis("off")
plt.title("Augmented Image")
plt.imshow(final_images[0][:,:,54], cmap='gray')

plt.subplot(2, 2, 3).axis("off")
plt.title("Original Mask")
plt.imshow(masks[0][:,:,54])
          
plt.subplot(2, 2, 4).axis("off")
plt.title("Augmented Mask")
plt.imshow(final_masks[0][:,:,54])


# In[8]:


plot_all(images[0])


# In[9]:


# Check shapes & Splits into training/testing
x_train, x_test, y_train, y_test = train_test_split(final_images, final_masks, test_size=0.20, random_state=7)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


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
    
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    # s4, p4 = encoder_block(p3, 256)

    b1 = conv_block(p3, 512) #Bridge

    # d1 = decoder_block(b1, s4, 256)
    d2 = decoder_block(b1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

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


model = build_unet((128, 128, 128, 1), n_classes=2)
model.compile(optimizer = optim, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=METRICS)
print(model.summary())


# In[15]:


'''Checks'''
print("Input shape", model.input_shape)
print("Output shape", model.output_shape)
print("-------------------")


# In[ ]:


history=model.fit(x_train, y_train,
        validation_data=(x_test, y_test),
        batch_size=1,
        epochs=100,
        shuffle=True,
        verbose=1)     


# In[ ]:


#Save model for future use
model.save('3D_UNet_no_patch_1.h5')


# In[32]:


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
            plt.ylim([0.8,1.1])
        else:
            plt.ylim([0,1.1])

        plt.legend();


# In[33]:


plot_metrics(history)


# In[8]:


#Load the pretrained model for testing and predictions. 
from tensorflow.keras.models import load_model
my_model = load_model('3D_UNet_no_patch_normalized.h5', compile=False)
#If you load a different model do not forget to preprocess accordingly. 


# In[9]:


''' Get the train and test sets after crop & resize '''
x_train_orig, x_test_orig, y_train_orig, y_test_orig = train_test_split(final_images, final_masks, test_size=0.20, random_state=7)
print(x_train_orig.shape, y_train_orig.shape)
print(x_test_orig.shape, y_test_orig.shape)


# In[10]:


''' PREDICT on Training Set '''
img = np.expand_dims(x_train_orig, axis=-1)
print(img.shape)
ground_truth = y_train_orig
print(ground_truth.shape)

end = []
# Prediction on each individual image because i was getting
# a resource exhaust error when i tried to predict on the whole batch
for i in img:
    end.append(my_model.predict(np.expand_dims(i, axis=0)))
end = np.asarray(end)
end = np.squeeze(end)
print(end.shape)
train_prediction = np.argmax(end, axis=4)[:,:,:,:]
print(train_prediction.shape)


# In[11]:


''' MEAN IOU on Training Set '''
from tensorflow.keras.metrics import MeanIoU

n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes) 

gt1 = ground_truth.astype("int32")
IOU_keras.update_state(gt1, train_prediction)
print("Training Set:", IOU_keras.result().numpy())


# In[12]:


''' Training Set Per Pixel Basis '''
from sklearn.metrics import confusion_matrix
import seaborn as sns
y_train_matrix = confusion_matrix(np.asarray(gt1).flatten(), np.asarray(train_prediction).flatten())

ax= plt.subplot()
sns.heatmap(y_train_matrix, annot=True, fmt='d', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels'); 
ax.set_title('Train Set Per Pixel Basis'); 
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);


# In[13]:


''' PREDICT on Testing Set '''
# Image from Testing Set
img = np.expand_dims(x_test_orig, axis=-1)
end = []
for i in img:
    end.append(my_model.predict(np.expand_dims(i, axis=0)))

end = np.asarray(end)
end = np.squeeze(end)
print(end.shape)
test_prediction = np.argmax(end, axis=4)[:,:,:,:]
print(test_prediction.shape)

ground_truth = y_test_orig.astype("int32")
print(ground_truth.shape)
print(np.unique(ground_truth))


# In[14]:


''' MEAN IOU on Testing Set'''
from tensorflow.keras.metrics import MeanIoU

n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)  
gt2 = ground_truth
IOU_keras.update_state(gt2, test_prediction)
print("Testing Set:", IOU_keras.result().numpy())


# In[15]:


''' Test Set Per Pixel Based '''
y_test_matrix = confusion_matrix(np.asarray(gt2).flatten(), np.asarray(test_prediction).flatten())

ax= plt.subplot()
sns.heatmap(y_test_matrix, annot=True, fmt='d', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels'); 
ax.set_title('Test Set Per Pixel Basis'); 
ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1']);


# In[16]:


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


# In[17]:


plot_three(x_test_orig[3], y_test_orig[3], test_prediction[3])


# In[ ]:




