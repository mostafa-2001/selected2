#!/usr/bin/env python
# coding: utf-8

# In[134]:


# import the libraries as shown below
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


# In[135]:


IMAGE_SIZE = [224, 224]
train_path = 'D:/computer science/level3/semester 2/selected 2/selected2/Datasets/Train'
valid_path = 'D:/computer science/level3/semester 2/selected 2/selected2/Datasets/Test'


# In[136]:


# Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG
# Here we will be using imagenet weights

resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[137]:


# don't train existing weights
for layer in resnet.layers:
    layer.trainable = False


# In[138]:


# useful for getting number of output classes
folders = glob('D:/computer science/level3/semester 2/selected 2/selected2/Datasets/Train/*')


# In[139]:


folders


# In[140]:



# our layers - you can add more if you want
x = Flatten()(resnet.output)


# In[141]:


prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=resnet.input, outputs=prediction)


# In[142]:


model.summary()


# In[143]:


# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[109]:


# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[144]:


# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('D:/computer science/level3/semester 2/selected 2/selected2/Datasets/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


# In[146]:


test_set = test_datagen.flow_from_directory('D:/computer science/level3/semester 2/selected 2/selected2/Datasets/Test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[112]:


# fit the model
# Run the cell. It will take some time to execute
r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=50,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


# In[113]:


r.history


# In[114]:


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


# In[115]:


from tensorflow.keras.models import load_model

model.save('D:/computer science/level3/semester 2/selected 2/selected2/model_resnet50.h5')


# In[116]:


y_pred = model.predict(test_set)


# In[117]:


y_pred


# In[118]:


import numpy as np
y_pred = np.argmax(y_pred, axis=1)


# In[119]:


y_pred


# In[120]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.image as mpimg


# In[147]:


img1 = mpimg.imread("D:/computer science/level3/semester 2/selected 2/selected2/Datasets/Test/mercedes/29.jpg")
imgplot = plt.imshow(img1)
plt.show()


# In[148]:


MODEL_PATH ='D:/computer science/level3/semester 2/selected 2/selected2/model_resnet50.h5'
model = load_model(MODEL_PATH)


# In[149]:


img_path = "D:/computer science/level3/semester 2/selected 2/selected2/Datasets/Test/mercedes/29.jpg"
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x=x/255
x = np.expand_dims(x, axis=0)
preds = model.predict(x)
preds=np.argmax(preds, axis=1)
if preds==0:
    preds="The Car IS Ford"
elif preds==1:
    preds="The Car Is lamborghini"
elif preds==2:
    preds="The Car Is Mercedes"
print(preds)


# In[ ]:




