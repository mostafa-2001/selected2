#!/usr/bin/env python
# coding: utf-8

# In[41]:


# import the libraries as shown below
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
#from tensorflow.keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# In[42]:


# Loading all necessary libraries and modules
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


# In[43]:


IMAGE_SIZE = [224, 224]
train_path = 'D:/computer science/level3/semester 2/selected 2/selected2/Datasets/Train'
valid_path = 'D:/computer science/level3/semester 2/selected 2/selected2/Datasets/Test'


# In[44]:


vgg16 = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[45]:


orig = cv.imread("D:/computer science/level3/semester 2/selected 2/selected2/Datasets/Train/mercedes/17.jpg")

# Convert image to RGB from BGR (another way is to use "image = image[:, :, ::-1]" code)
orig = cv.cvtColor(orig, cv.COLOR_BGR2RGB)

# Resize image to 224x224 size
image = cv.resize(orig, (224, 224)).reshape(-1, 224, 224, 3)

# We need to preprocess imageto fulfill VGG16 requirements
image = preprocess_input(image)

# Extracting our features
features = vgg16.predict(image)

features.shape


# In[46]:


n_features = features.shape[-1]
fig = plt.figure(figsize = (17, 8))
gs = gridspec.GridSpec(1, 2, figure = fig)
sub_gs = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs[1])
ax1 = fig.add_subplot(gs[0])
ax1.imshow(orig)
for i in range(3):
    for j in range(3):
        ax2 = fig.add_subplot(sub_gs[i, j])
        plt.axis('off')        
        plt.imshow(features[0, :, :, np.random.randint(n_features)], cmap = 'gray')  


# In[49]:


# don't train existing weights
for layer in vgg16.layers:
    layer.trainable = False


# In[50]:


# useful for getting number of output classes
folders = glob('D:/computer science/level3/semester 2/selected 2/selected2/Datasets/Train/*')


# In[51]:


folders


# In[52]:



# our layers - you can add more if you want
x = Flatten()(vgg16.output)


# In[53]:


prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg16.input, outputs=prediction)


# In[54]:


model.summary()


# In[55]:


# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[56]:


# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Data Augmentation 
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[57]:


# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('D:/computer science/level3/semester 2/selected 2/selected2/Datasets/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


# In[58]:


test_set = test_datagen.flow_from_directory('D:/computer science/level3/semester 2/selected 2/selected2/Datasets/Test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[60]:


# fit the model
# Run the cell. It will take some time to execute
r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set),
)


# In[61]:


r.history


# In[62]:


# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend() 
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[99]:


from tensorflow.keras.models import load_model

model.save('D:/computer science/level3/semester 2/selected 2/selected2/model_VGG16.h5')


# In[64]:


y_pred = model.predict(test_set)


# In[65]:


y_pred


# In[66]:


import numpy as np
y_pred = np.argmax(y_pred, axis=1)


# In[67]:


y_pred


# In[68]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.image as mpimg


# In[97]:


img_path = "C:/Users/DELL/OneDrive/Desktop/download (5).jpg"
img1 = mpimg.imread(img_path)
imgplot = plt.imshow(img1)
plt.show()


# In[100]:


MODEL_PATH ='D:/computer science/level3/semester 2/selected 2/selected2/model_VGG16.h5'
model = load_model(MODEL_PATH)


# In[101]:


img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x=x/255
x = np.expand_dims(x, axis=0)
preds = model.predict(x)
preds=np.argmax(preds, axis=1)
if preds==0:
     preds="The Car Is Jeep"
elif preds==1:
     preds="The Car Is Volkswagen"
elif preds==2:
     preds="The Car Is Mercedes"

print(preds)


# In[ ]:




