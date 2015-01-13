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

struct detector_opt
{
	int nPerOct ;//number of scales per octave
	int nOctUp ; //number of up_sampled octaves to compute
	int shrink;  
	int smooth ;//radius for channel smoothing (using convTri)
	int nbins;  //number of orientation channels
	int binsize;//spatial bin size
	int nApprox;// number of approx
	Size diam ; //minimum image size for channel computation
	detector_opt()
	{
		nPerOct=8 ;
		nApprox=7;
		nOctUp=0 ;
		smooth =1;
		shrink=4;
		diam=Size(16,16) ;
		nbins=6;
		binsize=4;
	}
};
class feature_Pyramids
{
public:

	feature_Pyramids();

	~feature_Pyramids();

	void chnsPyramid(const Mat &img,  vector<vector<Mat> > &approxPyramid);

	void convTri(const Mat &src, Mat &dst, const Mat &Km);

	void computeChannels( const Mat &image,vector<Mat>& channels);

	void computeGradient(const Mat &img, Mat& grad, Mat& qangle);	

	void setParas (const  detector_opt &in_para ) ;

	const detector_opt &getParas() const;

  private:

	  detector_opt m_opt;

};
#endif

