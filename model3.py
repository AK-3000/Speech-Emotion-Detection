# IMPORT NECESSARY LIBRARIES
import librosa
import soundfile as sf
# %matplotlib inline
import matplotlib.pyplot as plt
import librosa.display
from IPython.display import Audio
import numpy as np
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
from sklearn.metrics import confusion_matrix
import IPython.display as ipd  # To play sound in the notebook
import os # interface with underlying OS that python is running on
import sys
import warnings
# ignore warnings 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

#read the RAVDESS_SPEECH Dataset
audio = "/content/gdrive/MyDrive/major project/ravdess_speech/"
actor_folders = os.listdir(audio) #list files in audio directory
actor_folders.sort()

#Extract information : Emotion, gender, Path and create Datasets

emotion = []
gender = []
path = []
for i in actor_folders:
  rv_audio_files = os.listdir(audio +i)

  for f in rv_audio_files:
    part = f.split('.')[0].split('-')
    print(f)
    emotion.append(int(part[2]))
    g = int(part[6])
    if g%2 == 0:
      g = "female"
    else:
      g = "male"
    gender.append(g)
    path.append(audio +i+ '/' + f)

len(path)

#Creating RAVDESS Speech Dataframe

audio_df = pd.DataFrame(emotion)
audio_df = audio_df.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'})
audio_df = pd.concat([audio_df, pd.DataFrame(gender),pd.DataFrame(path)],axis=1)
audio_df.columns = ['emotion','gender', 'path']
#audio_df = pd.concat([audio_df,pd.DataFrame(file_path, columns = ['path'])],axis=1)
pd.set_option('display.max_colwidth', -1)

#checking if the emotions are correctly labelled
audio_df.sample(20)

#Reading SAVEE Audio Files
sv_audio = "/content/gdrive/MyDrive/major project/savee/ALL/"
sv_audio_files= os.listdir(sv_audio)


#Extracting Emotions, gender, Path from SAVEE files to create dataframe
emotion2 = []
gender2 = []
path2 = []
for f in sv_audio_files:

  if f[3:4] == "a":
    emotion2.append("angry")
  elif f[3:4] == "h":
    emotion2.append("happy")
  elif f[3:4] == "f":
    emotion2.append("fear")
  elif f[3:4] == "d":
    emotion2.append("disgust")
  elif f[3:4] == "n":
    emotion2.append("neutral")
  elif f[3:5] == "sa":
    emotion2.append("sad")
  elif f[3:5] == "su":
    emotion2.append("surprise")
 
  gender2.append("male")
  path2.append(sv_audio + f)

#Creating SAVEE DataFrame

sv_df = pd.DataFrame(emotion2)
sv_df = pd.concat([sv_df, pd.DataFrame(gender2),pd.DataFrame(path2)],axis=1)
sv_df.columns = ['emotion', 'gender', 'path']

#Combining RAVDESS and SAVEE Datasets
audio_df=audio_df.append(sv_df, ignore_index=True)

#Bar Plot of the emotion distribution of RAVDESS SPEECH and SAVEE combined
audio_df.emotion.value_counts().plot(kind='bar')

#Saving dataframe to CSV
audio_df.to_csv('/content/gdrive/MyDrive/major project/audio.csv')

"""##Feature Extraction"""

audio_df2 = audio_df


#DATA AUGMENTATION

# FUNCTION TO ADD WHITE NOISE
def noise(x):
    noise_amp = 0.05*np.random.uniform()*np.amax(x)   
    x = x.astype('float64') + noise_amp * np.random.normal(size=x.shape[0])
    return x
    
# FUNCTION TO STRETCH THE SOUND
def stretch(x, rate=0.8):
    data = librosa.effects.time_stretch(x, rate)
    return data


# FUNCTION TO INCREASE SPEED AND PITCH 
def speedNpitch(x):
    length_change = np.random.uniform(low=0.8, high = 1)
    speed_fac = 1.4 / length_change 
    tmp = np.interp(np.arange(0,len(x),speed_fac),np.arange(0,len(x)),x)
    minlen = min(x.shape[0], tmp.shape[0])
    x *= 0
    x[0:minlen] = tmp[0:minlen]
    return x

x, sr = librosa.load("/content/gdrive/MyDrive/major project/ravdess_speech/Actor_01/03-01-07-02-01-01-01.wav")
x=speedNpitch(x)
Audio(data=x, rate=sr)

x, sr = librosa.load("/content/gdrive/MyDrive/major project/ravdess_speech/Actor_01/03-01-07-02-01-01-01.wav")
x=stretch(x)
Audio(data=x, rate=sr)

x, sr = librosa.load("/content/gdrive/MyDrive/major project/ravdess_speech/Actor_01/03-01-02-02-01-01-01.wav")
x=noise(x)
Audio(data=x, rate=sr)

x, sr = librosa.load("/content/gdrive/MyDrive/major project/ravdess_speech/Actor_01/03-01-04-02-01-01-01.wav")
plt.figure(figsize=(8, 4))
librosa.display.waveplot(x, sr=sr)
plt.title('Original - Male Sad')

x, sr = librosa.load("/content/gdrive/MyDrive/major project/ravdess_speech/Actor_01/03-01-04-02-01-01-01.wav")
plt.figure(figsize=(8, 4))
x=speedNpitch(x)
librosa.display.waveplot(x, sr=sr)
plt.title('Increasing speed&pitch- Male Sad')

## ADD NOISE AND USE FEATURE EXTRACTION
df_noise = pd.DataFrame(columns=['feature'])
counter=0
for index,path in enumerate(audio_df2.path):
    X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=3,sr=44100,offset=0.5)
        # noise 
    aug = noise(X)
    aug = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
    aug=np.mean(aug,axis=0)
    df_noise.loc[counter] = [aug]
    counter +=1
print(len(df_noise))

# ADD STRETCH AND USE FEATURE EXTRACTION ON AUDIO FILES
df_stretch=pd.DataFrame(columns=['feature'])
counter=0
for index,path in enumerate(audio_df.path):
    #get wave representation
    X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=3,sr=44100,offset=0.5)
        # stretch
    X = stretch(X)
    aug = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
    aug=np.mean(aug,axis=0)
    df_stretch.loc[counter] = [aug] 
    counter +=1
print(len(df_stretch))
df_stretch.head()

# ADD SPEED AND PITCH THEN USE FEATURE EXTRACTION
df_speedpitch = pd.DataFrame(columns=['feature'])
counter=0
for index,path in enumerate(audio_df.path):
    #get wave representation
    X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=3,sr=44100,offset=0.5)
        # speed pitch
    X = speedNpitch(X)
    aug = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
    aug=np.mean(aug,axis=0)
    df_speedpitch.loc[counter] = [aug] 
    counter +=1
print(len(df_speedpitch))
df_speedpitch.head()

# ADD CORRESPONDING EMOTION LABELS TO DF_NOISE['FEATURE']
labels = audio_df.emotion
path = audio_df.path
noise_df = pd.DataFrame(df_noise['feature'].values.tolist())
noise = pd.concat([labels,path,noise_df], axis=1)
noise = noise.rename(index=str, columns={"0": "label", "1": "path"})

# ADD CORRESPONDING EMOTION LABELS TO DF_STRETCH['FEATURE']

stretch_df = pd.DataFrame(df_stretch['feature'].values.tolist())
stretch_1 = pd.concat([labels,path, stretch_df,], axis=1)
stretch_1 = stretch_1.rename(index=str, columns={"0": "label", "1" : "path"})

speedpitch_df = pd.DataFrame(df_speedpitch['feature'].values.tolist())
speedpitch = pd.concat([labels,path, speedpitch_df,], axis=1)
speedpitch = speedpitch.rename(index=str, columns={"0": "label", "1" : "path"})

audio_df3 = pd.concat([stretch_1,speedpitch], ignore_index= True)
audio_df3 = audio_df3.fillna(0)

# ITERATE OVER ALL AUDIO FILES AND EXTRACT LOG MEL SPECTROGRAM MEAN VALUES INTO DF FOR MODELING 
df = pd.DataFrame(columns=['mel_spectrogram','mfcc','chroma','zcr'])

counter=0

for index,path in enumerate(audio_df3.path):
    X, sample_rate = librosa.load(path, res_type='kaiser_fast',duration=3,sr=44100,offset=0.5)
    
    #get the mel-scaled spectrogram 
    spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128,fmax=8000) 
    db_spec = librosa.power_to_db(spectrogram)
    #temporally average spectrogram
    log_spectrogram = np.mean(db_spec, axis = 0)
        
    # Mel-frequency cepstral coefficients (MFCCs)
    mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
    mfcc=np.mean(mfcc,axis=0)
    
    # compute chroma energy (pertains to 12 different pitch classes)
    chroma = librosa.feature.chroma_stft(y=X, sr=sample_rate)
    chroma = np.mean(chroma, axis = 0)


    # compute zero-crossing-rate (zcr:the zcr is the rate of sign changes along a signal i.e.m the rate at 
#     which the signal changes from positive to negative or back - separation of voiced andunvoiced speech.)
    zcr = librosa.feature.zero_crossing_rate(y=X)
    zcr = np.mean(zcr, axis= 0)
    
    df.loc[counter] = [log_spectrogram,mfcc,chroma, zcr]
    counter=counter+1   

print(len(df))
df.head()

zcr= pd.DataFrame(df['zcr'].values.tolist())
mel_spectrogram= pd.DataFrame(df['mel_spectrogram'].values.tolist())
chroma= pd.DataFrame(df['chroma'].values.tolist())
mfcc= pd.DataFrame(df['mfcc'].values.tolist())

print(zcr.shape)
print(chroma.shape)



df_combined = pd.concat([audio_df3,chroma,zcr],axis=1)
df_combined = df_combined.fillna(0)

# DROP PATH COLUMN FOR MODELING
df_combined.drop(columns='path',inplace=True)
#Shuffling the dataframe
df_combined = df_combined.sample(frac=1).reset_index(drop=True)

"""##CNN MODEL"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization, Dense
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

# TRAIN TEST SPLIT DATA
train,test = train_test_split(df_combined, test_size=0.25, random_state=42,
                               stratify=df_combined[['emotion']])

X_train = train.iloc[:,1:]
y_train = train.iloc[:,:1]

y_train.reset_index().groupby('emotion').nunique()

#separating features and labels
X_test = test.iloc[:,1:]
y_test = test.iloc[:,:1]

y_test.reset_index().groupby('emotion').nunique()

# NORMALIZE DATA
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

X_train.head()

#TURN DATA INTO ARRAYS FOR KERAS
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# ONE HOT ENCODE THE TARGET
# CNN REQUIRES INPUT AND OUTPUT ARE NUMBERS
lb = LabelEncoder()
y_train = to_categorical(lb.fit_transform(y_train))
y_test = to_categorical(lb.fit_transform(y_test))

# RESHAPE DATA TO INCLUDE 3D TENSOR 
X_train = X_train[:,:,np.newaxis]
X_test = X_test[:,:,np.newaxis]

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model

#BUILD 1D CNN LAYERS
model = tf.keras.Sequential()
model.add(layers.Conv1D(64, kernel_size=(10), activation='relu', input_shape=(X_train.shape[1],1)))
model.add(layers.Conv1D(128, kernel_size=(10),activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.05)))
model.add(layers.MaxPooling1D(pool_size=(8)))
model.add(layers.Dropout(0.6))
model.add(layers.Conv1D(128, kernel_size=(10),activation='relu'))
model.add(layers.MaxPooling1D(pool_size=(8)))
model.add(BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.6))
model.add(layers.Dense(8, activation='sigmoid'))
opt = keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
model.summary()

import tensorflow.keras as keras
import numpy as np

X_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)

# FIT MODEL AND USE CHECKPOINT TO SAVE BEST MODEL
checkpoint = ModelCheckpoint("/content/gdrive/MyDrive/major project/cnn_model/best_initial_model.hdf5", monitor='val_accuracy', verbose=1,
    save_best_only=True, mode='max', period=1, save_weights_only=True)

model_history=model.fit(X_train, y_train,batch_size=100, epochs=500, validation_data=(X_test, y_test),callbacks=[checkpoint])

# PLOT MODEL HISTORY OF ACCURACY AND LOSS OVER EPOCHS
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Initial_Model_Accuracy.png')
plt.show()
# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Initial_Model_loss.png')
plt.show()

model.save("/content/gdrive/MyDrive/major project/cnn_model/SER_cnn")

from tensorflow.keras.models import Sequential, save_model, load_model

filepath = "/content/gdrive/MyDrive/major project/cnn_model/"
save_model(model,filepath)

"""##MLP MODEL"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

# TRAIN TEST SPLIT DATA
train,test = train_test_split(df_combined, test_size=0.3, random_state=42,
                               stratify=df_combined[['emotion']])

X_train = train.iloc[:,1:]
y_train = train.iloc[:,:1]

y_train.reset_index().groupby('emotion').nunique()

#separating features and labels
X_test = test.iloc[:,1:]
y_test = test.iloc[:,:1]

y_test.reset_index().groupby('emotion').nunique()

# NORMALIZE DATA
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

print("[+] Number of training samples:", X_train.shape[0])
print("[+] Number of testing samples:", X_test.shape[0])
print("[+] Number of features:", X_train.shape[1])

# MLP Parameters
model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08, 
    'hidden_layer_sizes': (250,), 
    'learning_rate': 'adaptive', 
    'max_iter': 600,
    'random_state': 1 
    
}
mlp_model = MLPClassifier(**model_params)

mlp_model.fit(X_train,y_train.values.ravel())

y_pred=mlp_model.predict(X_test)

accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))

import pickle
pickle.dump(mlp_model, open("/content/gdrive/MyDrive/major project/mlp_model/mlp_classifier.model", "wb"))


cf_matrix = confusion_matrix(y_test, y_pred)
# Plot the confusion matrix.
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat=cm,             
                      show_absolute=True,
                      show_normed=True,
                      colorbar=True,
                      figsize=(8,8))
plt.show()

print('Best score for the training data:', mlp_model.best_score_, '\nusing', mlp_model.best_params_)


