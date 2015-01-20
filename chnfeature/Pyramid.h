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
	Size minDS ; //minimum image size for channel computation
	detector_opt()
	{
		nPerOct=8 ;
		nOctUp=0 ;
		shrink=4;
		smooth =1;
		minDS=Size(41,100) ;
		nbins=6;
		binsize=4;
		nApprox=7;
	}
};
class feature_Pyramids
{
public:

	feature_Pyramids();

	~feature_Pyramids();

	void chnsPyramid(const Mat &img, 
                    vector<vector<Mat> > &approxPyramid,
                    vector<double> &scales,
                    vector<double> &scalesh,
                    vector<double> &scalesw) const;

	void chnsPyramid(const Mat &img,  vector<vector<Mat> > &approxPyramid,vector<double> &scales) const;

	void convTri( const Mat &src, Mat &dst,const Mat &Km) const;

	void computeChannels( const Mat &image,vector<Mat>& channels) const;

	void computeGradient(const Mat &img, Mat& grad, Mat& qangle) const;	

	void setParas (const  detector_opt &in_para ) ;

	void compute_lambdas(const vector<Mat> &fold);

	const detector_opt &getParas() const;

  private:

	  detector_opt m_opt;
	  	
	  vector<double>lam;

};
Mat get_Km(int smooth );
#endif

