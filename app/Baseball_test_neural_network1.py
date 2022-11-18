#!/usr/bin/env python
# coding: utf-8

# In[88]:


from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import seaborn as sns
import matplotlib as plt
import tensorflow as tf
from pybaseball import statcast_pitcher
from pybaseball import playerid_lookup


# In[89]:
df_baseball = statcast_pitcher('2008-04-01','2022-11-18', player_id=playerid_lookup('Verlander','Justin'))



df_baseball = pd.read_csv("test_baseball.csv")
df_baseball.head()


# In[90]:


pitches = ["FF", "FS", "SL", "CH", "SI"]
df_baseball = df_baseball[df_baseball.pitch_type.isin(pitches) == True]


# In[91]:


df_baseball.fillna(0, inplace=True)
df_baseball


# In[92]:


df_baseball.apply(pd.to_numeric, errors='ignore')


# In[93]:


# FF is fastball    1
# SL is slider      2
# FS is splitter    3
# SI is sinker      4
# CH is changeup    5
# CU is curve       6

# if someone is on base, 1; 0 otherwise
# 1 for right handed
# 0 for left handed

# df.loc[df["gender"] == "male", "gender"] = 1
df_baseball.loc[df_baseball['pitch_type'] == "FF", "pitch_type"] = 1
df_baseball.loc[df_baseball['pitch_type'] == "SL", "pitch_type"] = 2
df_baseball.loc[df_baseball['pitch_type'] == "FS", "pitch_type"] = 3
df_baseball.loc[df_baseball['pitch_type'] == "SI", "pitch_type"] = 4
df_baseball.loc[df_baseball['pitch_type'] == "CH", "pitch_type"] = 5

df_baseball.loc[df_baseball['on_1b'] > 0, "on_1b"] = int(1)
df_baseball.loc[df_baseball['on_2b'] > 0, "on_2b"] = int(1)
df_baseball.loc[df_baseball['on_3b'] > 0, "on_3b"] = int(1)

df_baseball.loc[df_baseball['stand'] == "R", "stand"] = int(1)
df_baseball.loc[df_baseball['stand'] == "L", "stand"] = int(0)

#df_baseball.to_csv("updatedBaseball.csv", index=False)
df_baseball['pitch_type'].astype(str)
type(df_baseball['pitch_type'])
df_baseball.head(40)


# In[94]:


#df_baseball = pd.get_dummies(df_baseball)
df_baseball.head(40)


# In[95]:


from sklearn.model_selection import train_test_split
X=df_baseball.drop('pitch_type', axis=1)  # Features
#X.drop('zone', axis=1)
y=df_baseball.pitch_type  # Labels
#y.insert(3, "zone", df_baseball.zone, True)
#y["zone"] = df_baseball.zone


X.drop('zone', axis=1, inplace=True)
# Split dataset into training set and test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test
X.head()


# In[58]:


y = pd.get_dummies(y)


# In[80]:


y


# In[96]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test


# In[109]:


#y_train = pd.get_dummies(y_train)
#y_train
y_test = pd.get_dummies(y_test)
y_test


# In[102]:


X_train = np.asarray(X_train).astype('float32')
X_test = np.asarray(X_test).astype('float32')

y_train = np.array(y_train).astype('float32')
y_test = np.array(y_test).astype('float32')


# In[117]:


for i in X_test:
    print(i)


# In[103]:


# model 1 from https://github.com/danielajk99/pitch_predictor/blob/master/Pitch_Predictor_HardSoft.ipynb
model = Sequential()
from keras.optimizers import SGD
model.add(Dense(6, activation = 'tanh'))
model.add(Dense(5, activation = 'sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs = 10,
            batch_size = 1)


# In[104]:


# model 2 from https://github.com/ShafinH/Pitch-Prediction/blob/master/04_neural_net.ipynb
# compile the keras model
#model.compile(loss='category_crossentropy',
#              optimizer='sgd',
#              metrics=['accuracy'])


# creating the model
model = Sequential()
model.add(Dense(8, input_shape=(6,), activation='tanh'))
model.add(Dense(8, activation='tanh'))
model.add(Dense(5, activation='sigmoid'))


model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01,momentum=0.9),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])


# In[105]:


model.fit(X_train, y_train,epochs=4, batch_size=1, verbose=1)

y_pred = model.predict(X_test)


# In[113]:


for i in X_test:
    print(i)


# In[119]:


for i in y_pred:
    print(sum(i))


# In[110]:


score = model.evaluate(X_test, y_test,verbose=1)
print(score)


# In[ ]:





# In[ ]:




