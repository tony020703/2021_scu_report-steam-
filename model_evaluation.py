from tensorflow.keras import models,layers,preprocessing,optimizers,losses,metrics
from sklearn.metrics import f1_score,accuracy_score,precision_score
from model_trainning import datapreprocess,datacut,textembedding,choose_labels
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class evaluation():
    def __init__(self,ds_test,genres):
        predict_all=[]
        label_all=[]
        self.genres=genres
        for i in range(len(genres)): 
            model = keras.models.load_model("model/"+genres[i]+".h5")
            predict_list=[]
            labels_list=[]
            for features, labels in ds_test.map(choose_labels(i)):
                #predictions = model(features,training = False)
                predict_class = (model.predict(features) > 0.5).astype("int32")[:,0].tolist()
                predict_list.extend(predict_class)
                labels_list.extend(labels.numpy().tolist())
            predict_all.append(predict_list)
            label_all.append(labels_list)
        self.label = np.transpose(label_all)
        self.ptedict = np.transpose(predict_all)
    def prediction(self):
        return self.label,self.ptedict
    def evaluate_F1(self):
        genres_f1=[]
        labels=self.label
        predict=self.ptedict
        genres=self.genres
        for i in range(len(genres)):
            f1=f1_score(labels[:,i],predict[:,i])
            genres_f1.append(f1)
        return genres_f1,'F1_score'
    def evaluate_accuracy(self):
        genres_accuracy=[]
        labels=self.label
        predict=self.ptedict
        genres=self.genres
        for i in range(len(genres)):
            accuracy=accuracy_score(labels[:,i],predict[:,i])
            genres_accuracy.append(accuracy)
        return genres_accuracy,'Accuracy'
    def evaluate_precision(self):
        genres_precision=[]
        labels=self.label
        predict=self.ptedict
        genres=self.genres
        for i in range(len(genres)):
            precision=precision_score(labels[:,i],predict[:,i])
            genres_precision.append(precision)
        return genres_precision,'Precision'

def get_bar_Polyline(data,index,index_type,genres):
    game_type_count=[]
    for i, j in zip(genres,index):
        game_type_count.append((i,data[i].sum(),j))

    game_type_count.sort(key=lambda score: score[1])
    data_x=[x[0] for x in game_type_count]
    data_y1=[y[1] for y in game_type_count]
    data_y2=[y[2] for y in game_type_count]

    fig, ax1 = plt.subplots(figsize=(10,10))
    ax2 = ax1.twiny()
    ax1.set_xlim(0, 1)
    ax1.xaxis.set_ticks_position('top')
    ax1.barh(data_x,data_y2,label=index_type)
    ax2.set_xlim(0, len(data))
    ax2.xaxis.set_ticks_position('bottom')
    ax2.plot(data_y1,data_x,'o-', color='black', label="Total Games")
    
    fig.legend(loc=1, bbox_to_anchor=(1,0.1), bbox_transform=ax1.transAxes)
    plt.savefig('{}.png'.format(index_type))

if __name__=='__main__':
    df1=pd.read_csv('data//steam.csv')[['appid','name','release_date','genres']]
    df2=pd.read_csv('data//steam_description_data.csv').rename(columns={'steam_appid':'appid'})[['appid','detailed_description']]
    df=pd.merge(df1,df2,on='appid')

    preprocess=datapreprocess(df)
    df_data=preprocess.text_process()
    genres=preprocess.feature_get()

    data_cut=datacut(df_data)
    df_test,df_train=data_cut.time_cut()

    data_meb=textembedding(genre_train=genres)
    ds_test=data_meb.transform(df_test,shuffle_on=False)

    evaluation_data=evaluation(ds_test,genres)
    genres_f1,index_type_f1=evaluation_data.evaluate_F1()
    genres_precision,index_type_precision=evaluation_data.evaluate_precision()

    get_bar_Polyline(df_test,genres_f1,index_type_f1,genres)
    get_bar_Polyline(df_test,genres_precision,index_type_precision,genres)
