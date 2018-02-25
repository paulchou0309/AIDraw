## 简介
使用 quick draw 的[数据集](https://console.cloud.google.com/storage/browser/quickdraw_dataset/sketchrnn),实现手绘图画识别的功能.

## 模型
Google 使用了 sketch-rnn,不过其论文没有去细看,试着把数据转成 png 图片,使用 cnn 识别,在只有 6 种类别 (bee, bird, butterfly, flower, grass, rabbit) 的情况下效果还不错.

## 文件说明
1. trans_data.py: 用于将 quick draw 的数据转成 png 图片,quick draw 的数据描述的是笔画的点,每个点的信息为三维矢量(x, y, lifted),分别为该点的横纵坐标及笔是否从纸上移开.使用 matplotlib 将其转成 png 图片便于处理.由于 quick draw 的数据量太大,个人资源不够,故只取训练样本 1000,交叉验证样本及测试样本各 250.
2. transfer_learning.py: 迁移学习,直接使用预训练好的 Resnet18 模型,只是修改全连接层的输出类别数量.
3. prepare.sh: 使用 [Google colab](https://colab.research.google.com) 提供的免费 Tesla K80,只是一台虚拟机一次只能连 12 个小时,因此经常需要重新配置环境及准备数据.