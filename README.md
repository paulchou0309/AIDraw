## 简介
使用 quick draw 的[数据集](https://console.cloud.google.com/storage/browser/quickdraw_dataset/sketchrnn),实现手绘图画识别的功能.

## 模型
Google 使用了 sketch-rnn,不过其论文没有去细看,试着把数据转成 png 图片,使用 cnn 识别,在只有 8 种类别 (bee, bird, butterfly, flower, grass, leaf, rabbit, tree) 的情况下效果还不错, 在训练 20 epoches 之后准确率达到 0.8631.

## 文件说明
1. trans_data.py: 用于将 quick draw 的数据转成 png 图片,quick draw 的数据描述的是笔画的点,每个点的信息为三维矢量(x, y, lifted),分别为该点的横纵坐标及笔是否从纸上移开.使用 matplotlib 将其转成 png 图片便于处理.由于 quick draw 的数据量太大,个人资源不够,故只取训练样本 1000,交叉验证样本及测试样本各 250.运行命令为:
```py
python trans_data.py
```
2. transfer_learning.py: 迁移学习,直接使用预训练好的 Resnet18 模型.运行命令为:
```py
python transfer_learning.py
```
3. prepare.sh: 使用 [Google colab](https://colab.research.google.com) 提供的免费 Tesla K80,只是一台虚拟机一次只能连 12 个小时,因此经常需要重新配置环境及准备数据.每次重新连上带 GPU 的虚拟机时直接复制脚本内容上去运行环境就准备好了.
4. test_model.py: 使用测试样本测试模型的正确率,最新的正确率为 0.8631.运行命令为:
```py
python test_model.py
```
5. predict.py: 使用训练好的模型对手绘图案进行预测,将需要预测的图片放置在 ${path}/predict/img/ 目录下, 则执行命令: python predict.py {path}, 会以 json 数组形式按预测概率降序输出概率最高的 6 个类别到控制台, 如: ["leaf", "grass", "bird", "butterfly", "rabbit", "tree"].


**我的运行目录结构:**
```
.
├── npz_data
│   ├── bee.npz
│   ├── bird.npz
│   ├── butterfly.npz
│   ├── download.sh
│   ├── flower.npz
│   ├── grass.npz
│   └── rabbit.npz
├── png_data
|   ├── test
|   |   ├── bee
|   |   |   ├── 1.png
|   |   |   └── 2.png
|   |   ├── bird
|   |   ├── ...
|   |   └── rabbit
│   ├── train
│   └── valid
├── predict
│   └── img
│       └── 2600.png
├── predict.py
├── prepare.sh
├── README.md
├── requirements.txt
├── test_model.py
├── trans_data.py
├── transfer_learning.py
└── forest.pkl
数据转换 (npz -> png) 需要的文件及文件夹有: npz_data, trans_data.py
执行训练需要的文件及文件夹有: png_data, transfer_learning.py, forest.pkl(可选).
测试模型准确率需要的文件及文件夹有: png_data, test_model.py, forest.pkl
执行预测需要的文件及文件夹有: predict.py, forest.pkl, predict.
```
**运行环境**: Python3.6.4 的虚拟环境,依赖见 requirements.txt