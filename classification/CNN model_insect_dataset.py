#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import glob
import cv2

import os

#TRAINGING SET:


# In[9]:


def getImagesfromXie(no_of_classes):
    insect_images = []
    labels = []
    i = 0
    for insect_dir_path in glob.glob("/content/sample_data/Xie dataset/*"):
        insect_label = insect_dir_path.split("/")[-1]
        if no_of_classes == i:
            break
        for image_path in glob.glob(os.path.join(insect_dir_path, "*.jpg")):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)

            image = cv2.resize(image, (64, 64)) 
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            insect_images.append(image)
            labels.append(insect_label)
        i = i + 1

    insect_images = np.array(insect_images)
    labels = np.array(labels)
    label_to_id_dict = {v: i for i, v in enumerate(np.unique(labels))}
    id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
    id_to_label_dict


    label_ids = np.array([label_to_id_dict[x] for x in labels])
    insect_images.shape, label_ids.shape, labels.shape
    
    print ("Total Images : " +  str(insect_images.shape[0]))
    return insect_images, label_ids


# In[10]:


insects_5_classes, labels_5_classes = getImagesfromXie(no_of_classes=5)
insects_10_classes, labels_10_classes = getImagesfromXie(no_of_classes=10)
insects_16_classes, labels_16_classes = getImagesfromXie(no_of_classes=16)
insects_24_classes, labels_24_classes = getImagesfromXie(no_of_classes=24)


# In[12]:


#SETTING UP THE NEURAL NETWORK
def modelCNN(X_train, X_test, Y_train, Y_test, Y, numClasses):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    
    model_cnn = Sequential()
    # First convolutional layer, note the specification of shape
    model_cnn.add(Conv2D(32, kernel_size=(3,3),
                     activation='relu',
                     input_shape=(64, 64, 3)))

    #Second layer
    model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
    #model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
    model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

    #Third Layer
    model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
    #model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
    model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

    #Fourth layer
    model_cnn.add(Conv2D(128, (3, 3), activation='relu'))

    model_cnn.add(Conv2D(128, (3, 3), activation='relu'))
    model_cnn.add(MaxPooling2D(pool_size=(2, 2)))


    model_cnn.add(Dropout(0.25))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(128, activation='relu'))
    model_cnn.add(Dropout(0.5))
    model_cnn.add(Dense(numClasses, activation='softmax'))
    
    opt = keras.optimizers.Adam(lr=0.001)
    model_cnn.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])


    model_cnn.fit(X_train, Y_train,
              batch_size=64,
              epochs=50,
              verbose=1,
              validation_data=(X_test, Y_test))

    predict = model_cnn.predict(X_test, batch_size=1)
    y = np.argmax(predict, axis=1)

    print('Accuracy Score :',accuracy_score(Y, y))
   
    return accuracy_score(Y, y)    


# In[13]:


from sklearn.model_selection import KFold

def getKFoldCV(Images, Labels, numClasses):
    # KFold Cross Validation approach
    kf = KFold(n_splits=9,shuffle=True,random_state=1245)
    kf.split(Images)

    # Initialize the accuracy of the models to blank list. The accuracy of each model will be appended to this list
    accuracy_model = []

    # Iterate over each train-test split
    for train_index, test_index in kf.split(Images):
        # Split train-test
        X_train, X_test = Images[train_index], Images[test_index]
        Y_train, Y_test = Labels[train_index], Labels[test_index]
        Y=Y_test

        #Normalize color values to between 0 and 1
        X_train = X_train/255
        X_test = X_test/255

        #Make a flattened version for some of our models
        X_flat_train = X_train.reshape(X_train.shape[0], 64*64*3)
        X_flat_test = X_test.reshape(X_test.shape[0], 64*64*3)

        #One Hot Encode the Output
        Y_train = keras.utils.to_categorical(Y_train,numClasses)
        Y_test = keras.utils.to_categorical(Y_test,numClasses)

        # Train the model
        print('Original Sizes:', X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
        accuracy_model.append(modelCNN(X_train, X_test, Y_train, Y_test, Y, numClasses))
    
    return (accuracy_model)    


# #  9-Fold 5 Classes 

# In[ ]:


import time
start_time= time.time()
accurary_5Classes = getKFoldCV(insects_5_classes, labels_5_classes, 5)
end_time=time.time()
print(accurary_5Classes)
print("Total time taken {}".format(end_time-start_time)) 


# # 9-Fold 10 Classes 

# In[ ]:


import time
start_time= time.time()
accurary_10Classes = getKFoldCV(insects_10_classes, labels_10_classes, 10)
end_time=time.time()
print(accurary_10Classes)
print("Total time taken {}".format(end_time-start_time)) 


# # 9-Fold 16 Classes 

# In[ ]:


import time
start_time= time.time()
accurary_16Classes = getKFoldCV(insects_16_classes, labels_16_classes, 16)
end_time=time.time()
print(accurary_16Classes)
print("Total time taken {}".format(end_time-start_time)) 


# # 9-Fold 24 Classes 

# In[ ]:


import time
start_time= time.time()
accurary_24Classes = getKFoldCV(insects_24_classes, labels_24_classes, 24)
end_time=time.time()
print(accurary_24Classes)
print("Total time taken {}".format(end_time-start_time)) 

