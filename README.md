# Unet 使用tensorflow

python:3.5.2
tensorflow:1.4.0


## github地址：https://github.com/anxiangSir/unet

数据集：CCF大数据竞赛遥语义分割初赛复赛数据集
主要尝试的训练方法
使用全卷积网络Unet，更改输出通道
对数据进行旋转，翻转，模糊，噪声，光照等数据增强
加入L2正则化weight decay
将Unet最小特征图加入了全局金字塔池化