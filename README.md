# 巨量資料應用專題(CNN_文本分類)
數據來源於kaggle[來源連接](https://www.kaggle.com/nikdavis/steam-store-games?select=steam.csv)   

[資料型態的視覺化](https://github.com/tony020703/2021_scu_report-steam-/blob/main/data_visualization.ipynb)

這個實驗依照時間把數據分為訓練集和測試集，其中訓練集的數據在2018年10月(包含)之前，測試集的數據在2018年10月之後。  
在經過[model_trainning.py](https://github.com/tony020703/2021_scu_report-steam-/blob/main/model_trainning.py)利用CNN對每個遊戲的介紹進行分類訓練。

電腦配置為：  
Linux ubuntu 18.04  
GPU: NVIDIA Tesla T4  16G  
環境配置為：  
tensorflow-gpu==2.5.0  
keras==2.4.3  

驗證結果：
方法使用F1_score對測試集進行評估。  
經過[model_evaluation.py](https://github.com/tony020703/2021_scu_report-steam-/blob/main/model_evaluation.py)把結果得出的F1_Score視覺化。  
<img src="https://github.com/tony020703/2021_scu_report-steam-/blob/main/F1_score.png" width="500">
  
結果：  
從測試集的結果看出，在數據分佈不平衡的標籤中遊戲數量10以上的標籤能部分找到目標標籤，但在遊戲數量10以下的標籤未能有效分辨出來，在熱門標籤中不少的F1_Score也有0.5以上的表現，Early Access和Free to Play因為標籤性質比較特殊，所以比較不容易從文字中找出分類。
