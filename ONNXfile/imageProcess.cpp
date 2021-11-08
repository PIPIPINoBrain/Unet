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
#include"imageProcess.h"

using namespace std;
using namespace cv;


//输入图像处理
imageProcess::imageProcess(){
	M_imageProcess = 1;
}

imageProcess::~imageProcess() {}

Mat imageProcess::imageNormalize(Mat image_input) {

	if (image_input.cols <= 0 && image_input.rows <= 0 && image_input.channels() != 3)
	{
		cout << "Mat error" << endl;
		exit(-1);
	}

	float mean[] = { 0.485,0.456,0.406 };
	float std[] = { 0.229,0.224,0.225 };

	image_input.convertTo(image_input, CV_32FC3);


	Mat r(image_input.rows, image_input.cols, CV_32FC1);
	Mat g(image_input.rows, image_input.cols, CV_32FC1);
	Mat b(image_input.rows, image_input.cols, CV_32FC1);

	Mat out[3] = { r,g,b };
	split(image_input, out);

	r = (out[0] / 255 - mean[0]) / std[0];
	g = (out[1] / 255 - mean[1]) / std[1];
	b = (out[2] / 255 - mean[2]) / std[2];

	Mat channels1[] = { r,g,b };
	Mat img_nor(image_input.rows, image_input.cols, CV_32FC3);
	merge(channels1, 3, img_nor);

	return img_nor;
}

Mat imageProcess::maxvalueFindindex(Mat prediction) {

	
	const int dims = prediction.size[1];
	const int rows = prediction.size[2];
	const int cols = prediction.size[3];


	Mat M_index(rows, cols, CV_8UC1);                              //存储每个像素位置中dims像素最大值下标dim
	Mat M_maxval(rows, cols, CV_32FC1);                            //存储每个像素对应的最大值


	for (int dim = 0; dim < dims; dim++)
	{
		for (int row = 0; row < rows; row++)
		{
			float* ptrScore = prediction.ptr<float>(0, dim, row);
			uchar* ptrM_index = M_index.ptr<uchar>(row);
			float* ptrM_maxval = M_maxval.ptr<float>(row);

			for (int col = 0; col < cols; col++)
			{

				if (ptrScore[col] > ptrM_maxval[col])
				{
					ptrM_maxval[col] = ptrScore[col];
					ptrM_index[col] = (uchar)dim;
				}

			}
		}
	}
	return M_index;
}

Mat imageProcess::colorAllocate(Mat indextensor) {

	if (indextensor.cols <= 0 && indextensor.rows <= 0 && indextensor.channels() != 1)
	{
		cout << "Mat error" << endl;
		exit(-1);
	} 
	int rows = indextensor.rows;                                                                   //获取row，col
	int cols = indextensor.cols;

	vector<Vec3b> labels_color = { { 0,0,0 },{ 0,0,128 },{ 0,128,0 },{ 0,128,128 },{ 128,0,0 } };  // indextensor分类的对应颜色

	Mat result = Mat::zeros(rows, cols, CV_8UC3);                                                  //颜色填充图像

	for (int row = 0; row < rows; row++)
	{
		const uchar* ptrindextensor = indextensor.ptr<uchar>(row);

		Vec3b* ptrColor = result.ptr<Vec3b>(row);

		for (int col = 0; col < cols; col++)
		{
			if (ptrindextensor[col] == 0)
			{
				ptrColor[col] = labels_color[0];
			}
			else if (ptrindextensor[col] == 1)
			{
				ptrColor[col] = labels_color[1];
			}
			else if (ptrindextensor[col] == 2)
			{
				ptrColor[col] = labels_color[2];
			}
			else if (ptrindextensor[col] == 3)
			{
				ptrColor[col] = labels_color[3];
			}
			else
			{
				ptrColor[col] = labels_color[4];
			}
		}
	}

	return result;

}
