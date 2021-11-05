#all writen by mr ma
######################################


import cv2
import torch.onnx
import onnx
from modeling.unet import *
from onnx import helper

#path_in 存放需要转换为onnx的输入文件
#path_out 存放输出的onnx文件的位置
def onnxEXPORT(path_in, path_out):
    model = Unet()                                                  #模型
    torch_model = torch.load(path_in, map_location='cpu')           #权重文件
    model.load_state_dict(torch_model['state_dict'])


    #输入数据
    batch_size = 1
    input_shape = (3, 512, 512)
    model.eval()
    input = torch.randn(batch_size, *input_shape)

    #数据转换
    torch.onnx.export(model,
                      input,
                      path_out,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={"input": {0: "batch_size"},
                                    "output": {0: "batch_size"}
                                    })
    return 0

#检验是否转换成功 1 成功，0失败
#path 是输出的onnx文件位置
def verify(path):
    onnx_model = onnx.load(path)
    flag_check = onnx.checker.check_model(onnx_model)
    output = onnx_model.graph.output
    print(output)                   #打印层，输入输出维度
    print('errorCheck: {}'.format(flag_check))    #检查模型完整性
    return 0



#输入一张图像检验
def one_pictureTest(path_onnx, path_image):
    Net = cv2.dnn.readNetFromONNX(path_onnx)
    image = cv2.imread(path_image, 1)
    img = cv2.dnn.blobFromImage(image,scalefactor=1,size=(512,512),mean=[0.485,0.456,0.406],swapRB=True)     #归一化,调整通道

    Net.setInput(img)
    out = Net.forward()
    print(out)
    return out


if __name__ =='__main__':
    path_tar = r'.\pth_file\model_best.pth.tar'
    path_onnx = r'.\pth_file\onnx_out.onnx'
    path_image = r'C:\Users\i\Desktop\222.jpg'
    onnxEXPORT(path_tar, path_onnx)
    verify(path_onnx)
    one_pictureTest(path_onnx, path_image)
    print('输出onnx文件成功')






