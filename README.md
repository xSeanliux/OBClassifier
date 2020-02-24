# OBClassifier

## Introduction 
一個試著將所輸入的（`OB統整_正規化.csv`）資料輸入一個PyTorch的神經網路中，並嘗試判斷一顆恆星是否為懸臂星的工具
* 資料提供：洪辰亞 @ CKHS
* 程式撰寫：劉至軒 @ CKHS

## 用之前要做什麼事？
應該只有需要PyTorch，確定自己有就好。

## 怎麼用？
### preprocessing.py 
將`OB統整_正規化.csv`內的資料讀出並以Python好讀的方式（`np.array`）存入`data.pkl`檔案內的工具程式。
### dataLoader.py
不用管他，反正就是將資料的存取包起來方便PyTorch使用的工具程式。
### model.py 
這就是類神經網路的模型放的地方，模型如下：
```python        
Linear(14, 256) #input layer 
ReLU
Linear(256, 256) #hidden layer
ReLU
Linear(256, 2) #output layer
```
很陽春的，但是只有十四個欄位還要我怎麼樣= =
### train.py 
只要在`cmd`裡面執行`python train.py`就可以跑了！會開始訓練，每次都從`./training_weights.ckpt`拿上次訓練的結果繼續跑，每一千個Epoch會儲存一次，如果按`Ctrl+C`退出也會嘗試儲存。目前已經訓練了好幾千個Epoch了，不需要再訓練了（否則怕會Overfitting就不好了）

如果進行訓練的時候想要看到進度（Loss對時間），那還可以再開一個`cmd`的視窗，到這個資料夾後執行`tensorboard --logdir runs`，他會出現一個網址（ex.`http://localhost:6006`），去那邊就可以看到Loss對時間的圖了！
