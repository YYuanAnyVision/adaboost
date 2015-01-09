#ifndef PYRAMID_H
#define PYRAMID_H

#include <iostream>
#include <fstream> 
#include <cmath>
#include <vector>
#include <typeinfo>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp" 
#include <cv.h>
#include <cxcore.h> 
#include <cvaux.h>

using namespace std;
using namespace cv;

class feature_Pyramids
{
public:
	feature_Pyramids();
	~feature_Pyramids();
	struct detector_opt
	{
		int nPerOct ;//number of scales per octave
		int nOctUp ; //number of upsampled octaves to compute
		int shrink;  
		int smooth ;//radius for channel smoothing (using convTri)
		Size diam ; //minimum image size for channel computation
		int nbins;  //number of orientation channels
		int binsize;//spatial bin size
		int nApprox;// number of approx
	}opt;

	void chnsPyramid(Mat img,  vector<vector<Mat> > &approxPyramid,detector_opt opt);

private:
	void convTri(Mat src, Mat &dst,Mat Km);
	void computeChannels(Mat image,vector<Mat>& channels,Size paddingTL, Size paddingBR,int nbins,int binsize);
	void computeGradient(const Mat img, Mat& grad, Mat& qangle,
		Size paddingTL, Size paddingBR,int nbins);	
};
#endif

