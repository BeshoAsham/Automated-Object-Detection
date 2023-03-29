#!/usr/bin/env python
# coding: utf-8

# In[20]:


import cv2
import os
import numpy as np
from sklearn import metrics
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from keras.applications.vgg16 import VGG16
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support


# In[6]:


data_path = r"/train"
img_size=256             
counter=0            
X=[]
Y=[]
 
categories=os.listdir(data_path)
 
for category in categories:                                                            # this loop to know how many images in categories
    folder_path=os.path.join(data_path,category)                                       # make folder empty has the same path for dataset
    img_names=os.listdir(folder_path)                                                  # put each image in this folder
 
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        fullpath=os.path.join(data_path,category,img_name)
        try:
            img = cv2.imread(fullpath, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (img_size,img_size))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            X.append(img)
            Y.append(category)
            counter+=1
            print("preprocessing Image Number==> ",counter)
        except:
            print("Error in ==> ",counter)
imgs=np.array(X)
lbls=np.array(Y)
del X
del Y
 


# In[7]:


#Label Encoding
le = preprocessing.LabelEncoder()
le.fit(lbls)
lbls_encoded = le.transform(lbls)
 
#Train and Test Split
train_x, test_x,train_y, test_y = train_test_split(imgs,lbls_encoded,test_size=0.1)
 
#Normalization
train_x, test_x = train_x / 255.0,  test_x / 255.0
 
 


# In[8]:


from keras.applications import VGG16
#Feature Extraction
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size,img_size, 3))
 
for layer in VGG_model.layers:
    layer.trainable = False
 
VGG_model.summary()  
feature_extractor=VGG_model.predict(train_x)
features = feature_extractor.reshape(feature_extractor.shape[0], -1)
ann_features_train = features
feature_extractor_test=VGG_model.predict(test_x)
ann_features_test = feature_extractor_test.reshape(feature_extractor_test.shape[0], -1)
 


# In[16]:


import keras
#ANN
#val_loss_(dec)_val_acc_(inc)
model = keras.models.Sequential([
   keras.layers.Flatten(),
   keras.layers.Dense(256, activation="relu"),
   keras.layers.Dense(128, activation="relu"),
   keras.layers.Dense(7, activation="softmax")]) # change number based on output classes
model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])
ann=model.fit(ann_features_train, train_y, epochs=20,validation_data=(ann_features_test, test_y))
y_preds = model.predict(ann_features_test).argmax(axis=1)
Accuracy = accuracy_score(test_y,y_preds)
print("Accuracy :", Accuracy)


# In[17]:


test_y_Normal = le.inverse_transform(test_y)


# In[21]:


from tkinter import *
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
 
root = tk.Tk()
root.geometry("1500x1050")  # Size of the window 
root.resizable(width=False, height=False)
root.title('Object Detector')
root['background']='#222227' 
my_font1=('times', 18, 'bold')
my_font2=('times', 12, 'bold')
label = tk.Label(root,text='Upload Files & Detect',width=30,font=my_font1)
label.grid(row=1,column=1)
label.place(anchor = CENTER, relx = .5, rely = .025)
 
 
b1 = tk.Button(root, text='Upload Images', 
   width=20,command = lambda:upload_file())
b1.grid(row=2,column=1,pady=5)
b1.place(anchor = CENTER, relx = .5, rely = .070)
def upload_file():
    f_types = [('Jpg Files', '*.jpg'),
    ('PNG Files','*.png'),('Jpeg Files', '*.jpeg')]   # types of files to select 
    filename = tk.filedialog.askopenfilename(multiple=True,filetypes=f_types)
    col=1 # start from column 1
    row=3 # start from row 3 
    for pathgui in filename:
        img=Image.open(pathgui)# read the image file
        list_of_images = []
        img_preprocessed = cv2.imread(pathgui, cv2.IMREAD_COLOR)
        img_preprocessed = cv2.resize(img_preprocessed, (img_size,img_size))
        img_preprocessed = cv2.cvtColor(img_preprocessed, cv2.COLOR_RGB2BGR)
        list_of_images.append(img_preprocessed)
        arr = np.array(list_of_images)
        feature_extractor_input=VGG_model.predict(arr)
        features_input = feature_extractor_input.reshape(feature_extractor_input.shape[0], -1)

        prediction_input = model.predict(features_input).argmax() #edited
        prediction_input_Normal = le.inverse_transform([prediction_input]) #edited
        img=img.resize((144,144)) # new width & height
        img=ImageTk.PhotoImage(img)
        e1 =tk.Label(root)
        e1.grid(row=row,column=col,pady=100,padx=10)
        e1.image = img
        text_answer=prediction_input_Normal[0] #edited
        # text_answer=text_answer.tolist()
        l2 = tk.Label(root,text=text_answer,width=20,font=my_font2)  
        l2.grid(row=row+1,column=col,pady=0,padx=10)
        e1['image']=img # garbage collection
        if(col==7): # start new line after third column
            row=row+2# start wtih next row
            col=1    # start with first column
        else:       # within the same row 
            col=col+1 # increase to next column                                                                                 
root.mainloop()  # Keep the window open# your code goes here


# In[22]:


cm=confusion_matrix(test_y, y_preds)
print(cm)
print(classification_report(test_y, y_preds))
# drawing confusion matrix
sns.heatmap(cm, center = True)
plt.show()


# In[26]:


#Calculating ROC:  
#roc_curve(y_test, y_pred, pos_label=None, sample_weight=None,drop_intermediate=True)

fprValue, tprValue, thresholdsValue = roc_curve(test_y,y_preds,pos_label=6)
#print('fpr Value  : ', fprValue)
#print('tpr Value  : ', tprValue)
#print('thresholds Value  : ', thresholdsValue)

#Calculating Area Under the Curve AUC :  

fprValue2, tprValue2, thresholdsValue2 = roc_curve(test_y,y_preds,pos_label=6 )
AUCValue = auc(fprValue2, tprValue2)
print('AUC Value  : ', AUCValue)
plt.show()

plt.figure(figsize=(5, 5), dpi=100)
plt.plot(fprValue,tprValue, marker='.', label='ANN (auc = %0.2f)' % AUCValue)
plt.legend()
plt.show()


# In[27]:


print("Loss Curve :",ann.history['loss'])
plt.plot(ann.history['loss'])
plt.title('loss curve')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')
plt.show()


# In[ ]:




