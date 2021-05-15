from tensorflow.keras import models,layers,preprocessing,optimizers,losses,metrics
from sklearn.metrics import f1_score
from model_trainning import datapreprocess,datacut,textembedding,choose_labels
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt

class evaluation:
    optimizer = optimizers.Nadam()
    loss_func = losses.BinaryCrossentropy()

    train_loss = metrics.Mean(name='train_loss')
    train_metric = metrics.BinaryAccuracy(name='train_accuracy')

    valid_loss = metrics.Mean(name='valid_loss')
    valid_metric = metrics.BinaryAccuracy(name='valid_accuracy')

    def valid_step(self, model, features, labels):
        predictions = model(features,training = False)
        predict_class=(model.predict(features) > 0.5).astype("int32")[:,0]
        batch_loss = self.loss_func(labels, predictions)
        self.valid_loss.update_state(batch_loss)
        #self.valid_metric.update_state(labels.numpy(), predict_class)
        return labels.numpy().tolist(),predict_class.tolist()
    def evaluate_model(self, model,ds_valid):
        labels_list=[]
        predict_list=[]
        for features, labels in ds_valid:
            labels_x,predict_x=self.valid_step(model,features,labels)
            labels_list.extend(labels_x)
            predict_list.extend(predict_x)
        self.valid_metric.update_state(labels_list, predict_list)
        f1=f1_score(labels_list, predict_list)
        logs = 'Valid Loss:{},Valid Accuracy:{},F1_score:{}' 
        tf.print(tf.strings.format(logs,(self.valid_loss.result(),self.valid_metric.result(),f1)))
        self.valid_loss.reset_states()
        self.train_metric.reset_states()
        self.valid_metric.reset_states()
        return f1

def get_bar_f1(data,f1,genres):
    game_type_count=[]
    for i, j in zip(genres,f1):
        game_type_count.append((i,data[i].sum(),j))

    game_type_count.sort(key=lambda score: score[1])
    data_x=[x[0] for x in game_type_count]
    data_y1=[y[1] for y in game_type_count]
    data_y2=[y[2] for y in game_type_count]

    fig, ax1 = plt.subplots(figsize=(10,10))
    ax2 = ax1.twiny()
    ax1.set_xlim(0, 1)
    ax1.xaxis.set_ticks_position('top')
    ax1.barh(data_x,data_y2,label="F1_score")
    ax2.set_xlim(0, len(data))
    ax2.xaxis.set_ticks_position('bottom')
    ax2.plot(data_y1,data_x,'o-', color='black', label="Total Games")
    
    fig.legend(loc=1, bbox_to_anchor=(1,0.1), bbox_transform=ax1.transAxes)
    plt.savefig("F1_score.png")
    plt.show()

df1=pd.read_csv('data//steam.csv')[['appid','name','release_date','genres']]
df2=pd.read_csv('data//steam_description_data.csv').rename(columns={'steam_appid':'appid'})[['appid','detailed_description']]
df=pd.merge(df1,df2,on='appid')

preprocess=datapreprocess(df)
df_data=preprocess.text_process()
genres=preprocess.feature_get()

data_cut=datacut(df_data)
df_test,df_train=data_cut.time_cut()

data_meb=textembedding(genre_train=genres)
ds_test=data_meb.transform(df_test)

evaluation_data=evaluation()
genres_f1=[]
for i in range(len(genres)):
    print(genres[i])
    reconstructed_model = keras.models.load_model("model/"+genres[i]+".h5")
    f1=evaluation_data.evaluate_model(reconstructed_model,ds_test.map(choose_labels(i)))
    genres_f1.append(f1)

get_bar_f1(df_test,genres_f1,genres)