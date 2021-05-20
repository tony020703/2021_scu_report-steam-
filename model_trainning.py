import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import models,layers,preprocessing,optimizers,losses,metrics
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

nltk.download('wordnet')
MAX_WORDS=20000
MAX_LEN=1000
BATCH_SIZE=128
filter_sizes=[3,4,5]
filters=64
wordembedding_size=50

class datapreprocess:
    def __init__(self,predata):
        self.predata=predata
    def feature_get(self):
        genres=[]
        for i in self.predata['genres']:
            i=i.split(";")
            for j in i:
                if j not in genres:
                    genres.append(j)
        return genres
    def column_onehot(self):
        genres=self.feature_get()
        genres_count=pd.DataFrame(columns=['genres_count'],index=range(len(self.predata['appid'])))
        game_type=pd.DataFrame(columns=genres,index=range(len(self.predata['appid'])))
        game=pd.concat([self.predata,genres_count,game_type],axis=1)
        
        game[['genres_count']]=game.genres.str.split(';').str.len()
        for i in game.columns[6:]:
            game[[i]]=game['genres'].apply(lambda x: 1 if i in x else 0)
        return game
    def text_process(self):
        appid,detailed_description=[],[]
        game=self.column_onehot()
        for i, j in zip(game.appid, game.detailed_description):
            content=j
            soup=BeautifulSoup(content,features='lxml')
            text=soup.get_text().replace("\n", " ").replace("\t", " ").replace("\r", " ").lower()
            text=re.sub(r'(http.* |http.*$)',' ',text)
            text=re.sub(r'\W+',' ', text)
            lemmatizer = WordNetLemmatizer()
            text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
            appid.append(i)
            detailed_description.append(text)
        
        df_text_process=pd.DataFrame({'appid':appid,'detailed_description_after_process':detailed_description})
        df_game=pd.merge(game,df_text_process,on=['appid'])
        df_game['detailed_description']=df_game['detailed_description_after_process']
        del df_game['detailed_description_after_process']
        return df_game

class datacut:
    def __init__(self,data):
        self.data=data
    def time_cut(self,year=2018,month=10):
        df_test1=self.data[(self.data.release_date.str[0:4].astype(int)==year) & (self.data.release_date.str[5:7].astype(int)>month)]
        df_train1=self.data[(self.data.release_date.str[0:4].astype(int)==year) & (self.data.release_date.str[5:7].astype(int)<=month)]
        df_test2=self.data[self.data.release_date.str[0:4].astype(int)>year]
        df_train2=self.data[self.data.release_date.str[0:4].astype(int)<year]
        df_test=pd.concat([df_test1,df_test2])
        df_train=pd.concat([df_train1,df_train2])
        return (df_test,df_train)

class textembedding:
    vectorize_layer = TextVectorization(
        split = 'whitespace',
        max_tokens=MAX_WORDS-1,
        output_mode='int',
        output_sequence_length=MAX_LEN)
    def __init__(self,genre_train):
        self.genre_train=genre_train
    def fit_transform(self,df_train):
        ds_train_raw = tf.data.Dataset.from_tensor_slices((df_train['detailed_description'].values,df_train[self.genre_train].values)).shuffle(len(df_train)).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
        ds_text = ds_train_raw.map(lambda text,label: text)
        self.vectorize_layer.adapt(ds_text)
        ds_train = ds_train_raw.map(lambda text,label:(self.vectorize_layer(text),label)).prefetch(tf.data.experimental.AUTOTUNE)
        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
        model.add(self.vectorize_layer)
        filepath = "model/text_vector"
        model.save(filepath, save_format="tf")
        return ds_train
    def transform(self,df_test,shuffle_on=True):
        filepath = "model/text_vector"
        loaded_model = tf.keras.models.load_model(filepath)
        loaded_vectorizer = loaded_model.layers[0]
        if shuffle_on==True:
            ds_test_raw=tf.data.Dataset.from_tensor_slices((df_test['detailed_description'].values,df_test[self.genre_train].values)).shuffle(len(df_test)).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
        else:
            ds_test_raw=tf.data.Dataset.from_tensor_slices((df_test['detailed_description'].values,df_test[self.genre_train].values)).batch(len(df_test)).prefetch(tf.data.experimental.AUTOTUNE)
        ds_test = ds_test_raw.map(lambda text,label:(loaded_vectorizer(text),label)).prefetch(tf.data.experimental.AUTOTUNE)
        return ds_test


def convolution():
    inn = layers.Input(shape=(MAX_LEN, wordembedding_size, 1))
    cnns = []
    for size in filter_sizes:
        conv = layers.Conv2D(filters=filters, kernel_size=(size, wordembedding_size),
                            strides=1, padding='valid', activation='relu')(inn)
        pool = layers.MaxPool2D(pool_size=(MAX_LEN-size+1, 1), padding='valid')(conv)
        cnns.append(pool)
    outt = layers.concatenate(cnns)

    model = keras.Model(inputs=inn, outputs=outt)
    return model

def cnn_mulfilter():
    model = keras.Sequential([
        layers.Embedding(input_dim=MAX_WORDS, output_dim=wordembedding_size,input_length=MAX_LEN),
        layers.Reshape((MAX_LEN, wordembedding_size, 1)),
        convolution(),
        layers.Flatten(),
        layers.Dense(10, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')

    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                 loss=keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])
    return model

class choose_labels:
    def __init__(self, label_i):
        self.label_i=label_i
    def __call__(self, features, labels):
        labels=labels[:,self.label_i]
        return features, labels

if __name__ == '__main__':
    df1=pd.read_csv('data//steam.csv')[['appid','name','release_date','genres']]
    df2=pd.read_csv('data//steam_description_data.csv').rename(columns={'steam_appid':'appid'})[['appid','detailed_description']]
    df=pd.merge(df1,df2,on='appid')

    preprocess=datapreprocess(df)
    df_data=preprocess.text_process()
    genres=preprocess.feature_get()

    data_cut=datacut(df_data)
    df_test,df_train=data_cut.time_cut()

    data_meb=textembedding(genre_train=genres)
    if 'text_vector' in os.listdir('model'):
        ds_train=data_meb.transform(df_train)
    else:
        ds_train=data_meb.fit_transform(df_train)
    ds_test=data_meb.transform(df_test)

    for i in range(len(genres)):
        if genres[i] in list(map(lambda x: x[0:-3],os.listdir('model'))):
            pass
        else:
            print(genres[i])
            model=cnn_mulfilter()
            model.summary()
            callback = EarlyStopping(monitor="loss", patience=5, verbose=1, mode="auto")
            history = model.fit(ds_train.map(choose_labels(i)), epochs=100,callbacks=[callback],validation_data=(ds_test.map(choose_labels(i))))
            model.save("model/"+genres[i]+".h5")
