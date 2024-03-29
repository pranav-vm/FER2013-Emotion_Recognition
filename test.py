# load json and create model
from __future__ import division
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")


json_file= open('fer30.json', 'r')
model1 = json_file.read()
json_file.close()

model = model_from_json(model1)

# load weights into new model
model.load_weights("fer30.h5")
print("Loaded model from disk")
truey=[]
predy=[]

x = np.load('./modXtest.npy')
y = np.load('./modytest.npy')
yhat= model.predict(x)
yh = yhat.tolist()
yt = y.tolist()
count = 0

for i in range(len(y)):
    yy = max(yh[i])
    yyt = max(yt[i])
    predy.append(yh[i].index(yy))
    truey.append(yt[i].index(yyt))
    if(yh[i].index(yy)== yt[i].index(yyt)):
        count+=1
acc = (count/len(y))*100

#saving values for confusion matrix and analysis
np.save('truey', truey)
np.save('predy', predy)
print("Predicted and true label values saved")
print("Accuracy on test set :"+str(acc)+"%")