#include<iostream>
#include<vector>
#include<ctime>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/cudabgsegm.hpp"
#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include "imageProcess.h"

using namespace std;
using namespace cv;

int main()
{
	int count = cuda::getCudaEnabledDeviceCount();                     //可用Gpu数量
	printf("GPU Device Count : %d \n", count);

	clock_t start_0 = clock();                                        //时间记录

	cv::dnn::Net net = dnn::readNetFromONNX("./unet.onnx");      //加载模型
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);               //模型加速,注意加速在加载后边，否则无法加速该模型
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	cout << "-----------------------------------" << endl;
	cout << "NET ACCELERATE FINISHED" << endl;

	clock_t end_0 = clock();
	double outtime = (double)(end_0 - start_0) / CLOCKS_PER_SEC;   
	cout << "load onnx cost： " << outtime << "s" << endl;                           //输出加载模型花费时间


	String path_images = ".\\test";
	vector<String>src_test;
	glob(path_images, src_test, false);   //将文件夹路径下的所有图片路径保存到src_test中

	if (src_test.size() == 0)
	{
		printf("error!\n");              //检查路径
		exit(1);
	}
	cout << "该路径下有" << src_test.size() << "个文件" << endl;

	imageProcess imgProcess;

	for (int i = 0; i < src_test.size(); i++)
	{
		clock_t start = clock();

		Mat image = imread(src_test[i]);

		image.convertTo(image, CV_32FC3);
		cvtColor(image, image, COLOR_BGR2RGB);
		resize(image, image, Size(640, 512));	                        //输入大小
															//cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);             //归一化
		Mat image_new = imgProcess.imageNormalize(image);    //img=(img-mean)/std

		Mat blob = dnn::blobFromImage(image_new);  //预处理 交换维度+加一维



		net.setInput(blob);

		Mat predict = net.forward(); // 推理出结果

		Mat indextensor = imgProcess.maxvalueFindindex(predict);

		Mat result = imgProcess.imageNormalize(indextensor);

		clock_t end = clock();

		double outtime = (double)(end - start) / CLOCKS_PER_SEC;
		cout <<"第" << i <<"张图片花费： " << outtime << 's' <<endl;

		string sub2 = src_test[i].substr(8); //从下标为3开始截取长度为5位

		imwrite(".\\out\\" + sub2, result);
	}



}