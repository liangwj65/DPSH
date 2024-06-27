# DPSH算法的python实现

# 实验环境
实验平台是由课程提供的服务器，在一张RTX 3090型号的GPU上实现
```
python==3.10.8  torchvision==0.18  pytorch==2.3.1 cuda==12.1  
```

# 数据集
数据集CIFAR-10和NUS-WIDE-21的划分放在/data下
- CIFAR-10:由于原论文为matlab实现，先后运行DataPrepare.m以及SaveFig.m来处理CIFAR-10数据集的matlab版本。脚本实现如下划分：对于每个类，随机选择 1,000 张图像作为查询集，其余图像作为检索数据库。然后从检索数据库中抽取每类 500 张图像（共 5,000 张）作为未标记的训练集。
- NUS-WIDE-21:数据集的划分来自 DeepHash 库提供的样例，具体划分如下：每个类选取 100 个样本作为查询集，剩下样本作为数据库，每个类别选取 500 张图像作为训练集。在/data/NUS-WIDE中直接存放了相应数据集的索引以及one-hot编码。

# 如何运行    
```
python DPSH_CIFAR_10_demo.py
```

NUS-WIDE-m is different from  NUS-WIDE, so i made a distinction.  

269,648 images in NUS-WIDE , and 195834 images which are associated with 21 most frequent concepts.     

NUS-WIDE-m has 223,496 images,and  NUS-WIDE-m  is used in [HashNet(ICCV2017)](http://openaccess.thecvf.com/content_ICCV_2017/papers/Cao_HashNet_Deep_Learning_ICCV_2017_paper.pdf) and code [HashNet caffe and pytorch](https://github.com/thuml/HashNet)    

download [mirflickr](https://press.liacs.nl/mirflickr/mirdownload.html) , and use ./data/mirflickr/code.py to randomly select 1000 images as the test query set and 4000 images as the train set.

# Demo
- model imagenet_64bits_0.8824931967229359.zip
[Baidu Pan(Password: hash)](https://pan.baidu.com/s/1_BiOmeCRYx6cVTWeWq-O9g).
- matplotlib demo
```
cd demo
python demo.py   
```
<img src="https://github.com/swuxyj/DeepHash-pytorch/blob/master/demo/demo.png"  alt="Matplotlib Demo"/><br/>  

- Flask demo
```
cd demo/www
python app.py   
```
<img src="https://github.com/swuxyj/DeepHash-pytorch/blob/master/demo/www/flask.png"  alt="Flask Demo"/><br/>  
# Precision Recall Curve
I add some code in DSH.py:
```
        config["pr_curve_path"] = f"log/alexnet/DSH_{config['dataset']}_{bit}.json"
```
To get the Precision Recall Curve, you should copy the json path `    "DSH": "../log/alexnet/DSH_cifar10-1_48.json",` to precision_recall_curve.py  and run this file.  
```
cd utils
python precision_recall_curve.py   
```
<img src="https://github.com/swuxyj/DeepHash-pytorch/blob/master/utils/pr.png"  alt="Precision Recall Curve"/><br/>  

 
# Paper And Code
DPSH(IJCAI2016)  
paper [Feature Learning based Deep Supervised Hashing with Pairwise Labels](https://cs.nju.edu.cn/lwj/paper/IJCAI16_DPSH.pdf)   
code [DPSH-pytorch](https://cs.nju.edu.cn/_upload/tpl/00/ce/206/template206/code/DPSH.zip)

# Mean Average Precision,48 bits[AlexNet].


<table>
    <tr>
        <td>Algorithms</td><td>dataset</td><td>this impl.</td><td>paper</td>
    </tr>
    <tr>
        <td >DSH</td><td >cifar10-1</td> <td >0.800</td> <td >0.6755</td>
    </tr>
    <tr>
        <td ></td><td >nus_wide_21</td> <td >0.798</td> <td >0.5621</td>
    </tr>
    <tr>
        <td ></td><td >ms coco</td> <td >0.655</td> <td >-</td>
    </tr>
    <tr>
        <td ></td><td >imagenet</td> <td >0.576</td> <td >-</td>
    </tr>
    <tr>
        <td ></td><td >mirflickr</td> <td >0.735</td> <td >-</td>
    </tr>
    <tr>
        <td >DPSH</td><td >cifar10</td> <td >0.775</td> <td >0.757</td>
    </tr>
    <tr>
        <td ></td><td >nus_wide_21</td> <td >0.844</td> <td >0.851(0.812*)</td>
    </tr>
    <tr>
        <td ></td><td >imagenet</td> <td >0.502</td> <td >-</td>
    </tr>
    <tr>
        <td ></td><td >ms coco</td> <td >0.711</td> <td >-</td>
    </tr>
    <tr>
        <td ></td><td >voc2012</td> <td >0.608</td> <td >-</td>
    </tr>
    <tr>
        <td ></td><td >mirflickr</td> <td >0.781</td> <td >-</td>
    </tr>
    <tr>
        <td >HashNet</td><td >cifar10</td> <td >0.782</td> <td >-</td>
    </tr>
    <tr>
        <td ></td><td >nus wide81 m</td> <td >0.764</td> <td >0.7114</td>
    </tr>
    <tr>
        <td ></td><td >nus_wide_21</td> <td >0.830</td> <td >-</td>
    </tr>
    <tr>
        <td ></td><td >imagenet</td> <td >0.644</td> <td >0.6633</td>
    </tr>
    <tr>
        <td ></td><td >ms coco</td> <td >0.724</td> <td >0.7301</td>
    </tr>
    <tr>
        <td >DHN</td><td >cifar10</td> <td >0.781</td> <td >0.621</td>
    </tr>
    <tr>
        <td ></td><td >nus_wide_21</td> <td >0.841</td> <td >0.758</td>
    </tr>
    <tr>
        <td ></td><td >imagenet</td> <td >0.486</td> <td >-</td>
    </tr>
    <tr>
        <td ></td><td >ms coco</td> <td >0.712</td> <td >-</td>
    </tr>
    <tr>
        <td ></td><td >mirflickr</td> <td >0.775</td> <td >-</td>
    </tr>
    <tr>
        <td >DSDH</td><td >cifar10-1</td> <td >0.790</td> <td >0.820</td>
    </tr>
    <tr>
        <td ></td><td >nus_wide_21</td> <td >0.833</td> <td >0.829</td>
    </tr>
    <tr>
        <td ></td><td >imagenet</td> <td >0.300</td> <td >-</td>
    </tr>
    <tr>
        <td ></td><td >ms coco</td> <td >0.681</td> <td >-</td>
    </tr>
    <tr>
        <td ></td><td >mirflickr</td> <td >0.765</td> <td >-</td>
    </tr>
    <tr>
        <td >DTSH</td><td >cifar 10</td> <td >0.800</td> <td >0.774</td>
    </tr>
    <tr>
        <td ></td><td >nus_wide_21</td> <td >0.829</td> <td >0.824</td>
    </tr>
    <tr>
        <td ></td><td >ms coco</td> <td >0.760</td> <td >-</td>
    </tr>
    <tr>
        <td ></td><td >imagenet</td> <td >0.631</td> <td >-</td>
    </tr>
    <tr>
        <td ></td><td >mirflickr</td> <td >0.753</td> <td >-</td>
    </tr>
    <tr>
        <td >DFH</td><td >cifar10-1</td> <td >0.801</td> <td >0.844</td>
    </tr>
    <tr>
        <td ></td><td >nus_wide_21</td> <td >0.837</td> <td >0.842</td>
    </tr>
    <tr>
        <td ></td><td >ms coco</td> <td >0.717</td> <td >-</td>
    </tr>
    <tr>
        <td ></td><td >imagenet</td> <td >0.519</td> <td >0.747</td>
    </tr>
    <tr>
        <td ></td><td >mirflickr</td> <td >0.766</td> <td >-</td>
    </tr>
    <tr>
        <td >GreedyHash</td><td >cifar10-1</td> <td >0.817</td> <td >0.822</td>
    </tr>
    <tr>
        <td ></td><td >cifar10-2</td> <td >0.932</td> <td >0.944</td>
    </tr>
    <tr>
        <td ></td><td >imagenet</td> <td >0.678</td> <td >0.688</td>
    </tr>
    <tr>
        <td ></td><td >ms coco</td> <td >0.728</td> <td >-</td>
    </tr>
    <tr>
        <td ></td><td >nuswide_21</td> <td >0.793</td> <td >-</td>
    </tr>
    <tr>
        <td >ADSH</td><td >cifar10-1</td> <td >0.921</td> <td >0.9390</td>
    </tr>
    <tr>
        <td ></td><td >nuswide_21</td> <td >0.622</td> <td >0.9055</td>
    </tr>
    <tr>
        <td >CSQ(ResNet50,64bit)</td><td >coco</td> <td >0.883</td> <td >0.861</td>
    </tr>
    <tr>
        <td ></td><td >imagenet</td> <td >0.881</td> <td >0.873</td>
    </tr>
    <tr>
        <td ></td><td >nuswide_21_m</td> <td >0.844</td> <td >0.839</td>
    </tr>
</table>
Due to time constraints, I can't test many hyper-parameters  

