#!/usr/bin/env python
# coding: utf-8



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
import matplotlib.pyplot as plt



pitch_types = {
    'SL' : 'Slider',
    'CH' : 'Changeup',
    'CU' : 'Curveball',
    'FF' : 'Fastball',
    'FC' : 'Cutter',
    'SI' : 'Sinker',
    'FS' : 'Splitter'
}


df = pd.read_csv("savant_ohtani.csv")
df.head()


pitches = ["FF", "SL", "FS", "SI", "CH", "CU", 'FC']
df = df[df.pitch_type.isin(pitches) == True]
df.fillna(0, inplace=True)
df


df.apply(pd.to_numeric, errors='ignore')


# FF is fastball    1
# SL is slider      2
# FS is splitter    3
# SI is sinker      4
# CH is changeup    5
# CU is curve       6

# if someone is on base, 1; 0 otherwise
# 1 for right handed
# 0 for left handed

df.loc[df['pitch_type'] == "FF", "pitch_type"] = 1
df.loc[df['pitch_type'] == "SL", "pitch_type"] = 2
df.loc[df['pitch_type'] == "FS", "pitch_type"] = 3
df.loc[df['pitch_type'] == "SI", "pitch_type"] = 4
df.loc[df['pitch_type'] == "CH", "pitch_type"] = 5
df.loc[df['pitch_type'] == "CU", "pitch_type"] = 6
df.loc[df['pitch_type'] == "FC", "pitch_type"] = 7

df.loc[df['on_1b'] > 0, "on_1b"] = int(1)
df.loc[df['on_2b'] > 0, "on_2b"] = int(1)
df.loc[df['on_3b'] > 0, "on_3b"] = int(1)

df.loc[df['stand'] == "R", "stand"] = int(1)
df.loc[df['stand'] == "L", "stand"] = int(0)

df.loc[df['prev_desc'] == "force_out", "prev_desc"] = 1
df.loc[df['prev_desc'] == "called_strike", "prev_desc"] = 2
df.loc[df['prev_desc'] == "swinging_strike", "prev_desc"] = 3
df.loc[df['prev_desc'] == "ball", "prev_desc"] = 4
df.loc[df['prev_desc'] == "strikeout", "prev_desc"] = 5
df.loc[df['prev_desc'] == "foul", "prev_desc"] = 6
df.loc[df['prev_desc'] == "field_out", "prev_desc"] = 7
df.loc[df['prev_desc'] == "walk", "prev_desc"] = 8
df.loc[df['prev_desc'] == "single", "prev_desc"] = 9
df.loc[df['prev_desc'] == "hit_by_pitch", "prev_desc"] = 10
df.loc[df['prev_desc'] == "double", "prev_desc"] = 11
df.loc[df['prev_desc'] == "foul_tip", "prev_desc"] = 12
df.loc[df['prev_desc'] == "sac_fly", "prev_desc"] = 13
df.loc[df['prev_desc'] == "blocked_ball", "prev_desc"] = 14
df.loc[df['prev_desc'] == "grounded_into_double_play", "prev_desc"] = 15
df.loc[df['prev_desc'] == "double_play", "prev_desc"] = 16
df.loc[df['prev_desc'] == "triple", "prev_desc"] = 17
df.loc[df['prev_desc'] == "caught_stealing_2b", "prev_desc"] = 18
df.loc[df['prev_desc'] == "home_run", "prev_desc"] = 19
df.loc[df['prev_desc'] == "field_error", "prev_desc"] = 20
df.loc[df['prev_desc'] == "foul_bunt", "prev_desc"] = 21
df.loc[df['prev_desc'] == "other_out", "prev_desc"] = 22
df.loc[df['prev_desc'] == "swinging_strike_blocked", "prev_desc"] = 23
df.loc[df['prev_desc'] == "sac_bunt", "prev_desc"] = 24
df.loc[df['prev_desc'] == "missed_bunt", "prev_desc"] = 25
df.loc[df['prev_desc'] == "strikeout_double_play", "prev_desc"] = 26
df.loc[df['prev_desc'] == "fielders_choice_out", "prev_desc"] = 27
df.loc[df['prev_desc'] == "catcher_interf", "prev_desc"] = 28
df.loc[df['prev_desc'] == "hit_into_play", "prev_desc"] = 29

df['pitch_type'].astype(str)
type(df['pitch_type'])
df.head(40)


print(df.pitch_type.unique())


from sklearn.model_selection import train_test_split
X=df.drop('pitch_type', axis=1)  # Features
X.drop('Unnamed: 0', axis=1, inplace=True)
X.drop('events', axis=1, inplace=True)

y=df.pitch_type  # Labels


X.drop('zone', axis=1, inplace=True)
X.head()


col_list = list(X.columns)
a, b = col_list.index('balls'), col_list.index('strikes')
col_list[b], col_list[a] = col_list[a], col_list[b]
X = X[col_list]
X


y = pd.get_dummies(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test


y_test = pd.get_dummies(y_test)



X_train = np.asarray(X_train).astype('float32')
X_test = np.asarray(X_test).astype('float32')

y_train = np.array(y_train).astype('float32')
y_test = np.array(y_test).astype('float32')

# collects the loss and accuracy history to plot in a graph
loss_history = []
acc_history = []


# creating the model
model = Sequential()
model.add(Dense(128, input_shape=(len(X.columns),), activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(16, activation='tanh'))
model.add(Dense(8, activation='tanh'))
model.add(Dense(len(y.columns), activation='softmax'))


model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.9),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])


hist = model.fit(X_train, y_train,epochs=165, batch_size=5, verbose=1)
for i in range(len(hist.history['accuracy'])):
    acc_history.append(hist.history['accuracy'][i])
    
for i in range(len(hist.history['loss'])):
    loss_history.append(hist.history['loss'][i])

y_pred = model.predict(X_test)
plt.figure(figsize=(8,8))
plt.plot(loss_history,label='training_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()


plt.figure(figsize=(8,8))
plt.plot(acc_history,label='training_accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()


score = model.evaluate(X_test, y_test,verbose=1)
print(score)


def plottingNames(resultList):
    pitches = ["fastball", "slider", "splitter", "sinker", "curveball", "cutter"]
    plt.figure(figsize=(10,10))
    plt.barh(pitches, resultList)
    plt.show()


from random import randint

inputData = [[]]
desc = randint(1,24)
stand = randint(0,1)
strikes = randint(0,2)
balls = randint(0,3)
on1 = randint(0,1)
on2 = randint(0,1)
on3 = randint(0,1)
outUp = randint(0,2)
bat = randint(0,10)
fld = randint(0,10)

inputData[0].append(desc)
inputData[0].append(stand)
inputData[0].append(strikes)
inputData[0].append(balls)
inputData[0].append(on3)
inputData[0].append(on2)
inputData[0].append(on1)
inputData[0].append(outUp)
inputData[0].append(bat)
inputData[0].append(fld)


inputData = np.array(inputData).astype('float32')
score = model.predict(inputData)
result = score.tolist()

print(result[0])
plottingNames(result[0])

    

model.save('Ohtani')
