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
	int count = cuda::getCudaEnabledDeviceCount();                     //����Gpu����
	printf("GPU Device Count : %d \n", count);

	clock_t start_0 = clock();                                        //ʱ���¼

	cv::dnn::Net net = dnn::readNetFromONNX("./unet.onnx");      //����ģ��
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);               //ģ�ͼ���,ע������ڼ��غ�ߣ������޷����ٸ�ģ��
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	cout << "-----------------------------------" << endl;
	cout << "NET ACCELERATE FINISHED" << endl;

	clock_t end_0 = clock();
	double outtime = (double)(end_0 - start_0) / CLOCKS_PER_SEC;   
	cout << "load onnx cost�� " << outtime << "s" << endl;                           //�������ģ�ͻ���ʱ��


	String path_images = ".\\test";
	vector<String>src_test;
	glob(path_images, src_test, false);   //���ļ���·���µ�����ͼƬ·�����浽src_test��

	if (src_test.size() == 0)
	{
		printf("error!\n");              //���·��
		exit(1);
	}
	cout << "��·������" << src_test.size() << "���ļ�" << endl;

	imageProcess imgProcess;

	for (int i = 0; i < src_test.size(); i++)
	{
		clock_t start = clock();

		Mat image = imread(src_test[i]);

		image.convertTo(image, CV_32FC3);
		cvtColor(image, image, COLOR_BGR2RGB);
		resize(image, image, Size(640, 512));	                        //�����С
															//cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);             //��һ��
		Mat image_new = imgProcess.imageNormalize(image);    //img=(img-mean)/std

		Mat blob = dnn::blobFromImage(image_new);  //Ԥ���� ����ά��+��һά



		net.setInput(blob);

		Mat predict = net.forward(); // ��������

		Mat indextensor = imgProcess.maxvalueFindindex(predict);

		Mat result = imgProcess.imageNormalize(indextensor);

		clock_t end = clock();

		double outtime = (double)(end - start) / CLOCKS_PER_SEC;
		cout <<"��" << i <<"��ͼƬ���ѣ� " << outtime << 's' <<endl;

		string sub2 = src_test[i].substr(8); //���±�Ϊ3��ʼ��ȡ����Ϊ5λ

		imwrite(".\\out\\" + sub2, result);
	}



}