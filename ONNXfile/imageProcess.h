#pragma once

#ifndef __ImageProcess_H__
#define __PictureProcess_H__

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

using namespace std;
using namespace cv;


class imageProcess{
public:

	imageProcess();
	~imageProcess();

	Mat imageNormalize(Mat image_input);

	Mat maxvalueFindindex(Mat prediction);

	Mat colorAllocate(Mat indextensor);

private:
	int M_imageProcess;
};

#endif