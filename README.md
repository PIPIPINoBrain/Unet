#Unet
传统unet分割，按 照要求可以快速训练，生成onnx文件，并移植到opencv中

###环境
按照requirement配置，用的是旋转目标检测的环境，配好可以直接对旋转目标进行检测

###使用
Data中包含三个路径，Images为存放训练未标注图像位置，Segment_label为存放标注后图像位置,，
Segmentation存放了三个txt文件，train.txt存储了训练集图像文件名，val，trainval分别存储了测试集，总文件名

###输出
训练输出会在run中，并产生pth.tar文件最好权重吧，
将权重放到pth_file中，从onnxGet.py得到onnx文件，也可以直接改路径。

###测试pth.tar文件
链接：https://pan.baidu.com/s/1gP1EAeTXxj-MptxfLDDkQA 
提取码：mab6






项目做的是一个光板缺陷分割

