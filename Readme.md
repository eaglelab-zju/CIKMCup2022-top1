# CIKM 2022 AnalytiCup Competition: 联邦异质任务学习 - 冠军方案

## To run the code:
1.下载代码到本地
```
git clone https://github.com/cycl2018/DGLD-CIKM.git
```
2.配置conda虚拟环境
- 操作系统：Ubuntu 18.04.6 LTS
- GPU型号：NVIDIA GeForce RTX 3090
- 下载Conda pack打包环境（链接：https://pan.baidu.com/s/1NL9saBAx2KbKa1K3ncTT1g 提取码：hs9n），放置在~/downloads文件夹下
```
cd DGLD-CIKM
mkdir dgld-cikm
tar -xzf ~/downloads/cikmenv.tar.gz -C dgld-cikm
source dgld-cikm/bin/activate
```
3.下载数据集文件
- 从天池官网下载数据集压缩包，放置在~/downloads文件夹下
- 并解压到save/data/raw目录下，注意raw目录下数据文件存放格式如save/data/raw/Readme.md中所示！
```
unzip ~/downloads/CIKM22Competition.zip -d save/data
cp -r save/data/CIKM22Competition/* save/data/raw
```
4.运行程序复现结果，提交的文件位置为save/submit.csv
```
bash scripts/run.sh
```
## 算法介绍
### 联邦学习算法
- 考虑到各个客户任务以及属性的异质性，我们直接忽略异质信息，直接对各个客户端的图结构进行联邦学习，服务端与客户端共享同一个模型。具体模型采用[GraphMAE](https://arxiv.org/abs/2205.10803)，开源地址：https://github.com/THUDM/GraphMAE 其中encoder与decoder均采用gin，每个客户端所拥有的数据均忽略自带属性，将各个节点度的onehot编码作为节点属性进行训练。具体的服务端和客户端代码在Fed_learn/server_struct.py与Fed_learn/client_struct.py中。
- 联邦训练方式：按照我们的设置，已经将问题转换为了横向联邦。服务端与客户端之间通过参数共享的方式进行联邦训练。具体的形式为，1、各个客户端加载好本地的数据，并初始化模型。2、每一轮训练，服务端先随机打乱客户端的顺序，依次进行训练。3、第i个客户端训练前，服务端将模型参数发送给客户端i，客户端i更新参数，采用SGD的方式使用本地数据训练一个epoch，训练完成后将模型参数发回服务端，服务端更新参数。4、按照这种方式训练200个epoch。
- 我们保留最终训练出来的模型，并将该模型的encoder部分作为图结构的编码器，加入到各个客户端的个性化训练中。
### 客户端个性化训练
- 我们对每个客户端都单独进行个性化的设置进行最终的训练预测
- 在client1、2、5、8中使用了联邦学习出来的结构embedding，其他客户端均仅使用本地数据训练预测
- 模型代码：Fed_learn/client.py
### GINE模型
- 开源地址：https://github.com/awslabs/dgl-lifesci
- 利用了边属性的GIN模型
- 模型代码:models/GINE.py
### GINE_S
- 加入联邦训练的结构encoder后的GINE模型
- 具体结合方式为将encoder的节点embedding进行线性变换对齐GINE生成的embedding，叠加。
- 模型代码：models/GINE_S.py
### GraphMAE预训练
- 利用GraphMAE的方式对GINE模型预训练
- 模型代码：models/maepre.py
### GINE-bot
- GINE-bot全称为GINE and bag of tricks，其实现参考了ogb中的Leaderboards for Graph Property Prediction上的开源项目[YouGraph](https://github.com/PierreHao/YouGraph)
- 以GINE模型为backbone，并使用了node degree、virtual node、AdamW等trick
- 模型代码：models/GNN.py
- config中配置model选项命名为`GNN-dgl`

### 各个客户端个性化方案
|client|方案|
| --- | --- |
|client1|GINE_S|
|client2|GINE_S+focal loss+GraphMAE预训练+模型融合|
|client3|GINE+GraphMAE预训练|
|client4|GINE|
|client5|GINE_S+focal loss+模型融合|
|client6|GINE-bot+focal loss|
|client7|GINE+focal loss|
|client8|GINE_S|
|client9|GINE-bot|
|client10|GINE-bot|
|client11|GINE-bot|
|client12|GINE-bot|
|client13|GINE-bot|

