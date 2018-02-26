## 简介
使用 quick draw 的[数据集](https://console.cloud.google.com/storage/browser/quickdraw_dataset/sketchrnn),实现手绘图画识别的功能.

## 模型
Google 使用了 sketch-rnn,不过其论文没有去细看,试着把数据转成 png 图片,使用 cnn 识别,在只有 6 种类别 (bee, bird, butterfly, flower, grass, rabbit) 的情况下效果还不错, 在训练 10 epoches 之后准确率达到 88.75%.

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
4. test_model.py: 使用测试样本测试模型的正确率,最新的正确率为 88.75%.运行命令为:
```py
python test_model.py
```
5. predict.py: 使用训练好的模型对手绘图案进行预测,如当前目录下有图片 bee.png, 则执行命令: python predict.py bee.png, 若预测正确会输出: bee


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
├── predict.py
├── prepare.sh
├── README.md
├── test_model.py
├── trans_data.py
├── transfer_learning.py
└── transfer.pkl
```