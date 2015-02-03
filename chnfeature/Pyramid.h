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
	Size pad;
	detector_opt()
	{
		nPerOct=8 ;
		nOctUp=0 ;
		shrink=4;
		smooth =1;
		minDS=Size(41,100) ;
		pad=Size(12,16);
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
	void chnsPyramid(const Mat &img,  vector<vector<Mat> > &chns_Pyramid,vector<double> &scales) const;

	void convTri( const Mat &src, Mat &dst,const Mat &Km) const;        //1 opencv version

    //2 sse version, faster
	void convTri( const Mat &src,       // in :data, for color image, this is the first channel, and set dim =3
                  Mat &dst, 
                  int conv_size, 
                  int dim) const;       // in :dim, DO NOT SET dim=3 for gray image, and should make 3 channels continuous 

	void getscales(const Mat &img,vector<Size> &ap_size,vector<int> &real_scal,vector<double> &scales,vector<double> &scalesh,vector<double> &scalesw) const;

	void get_lambdas(vector<vector<Mat> > &chns_Pyramid,vector<double> &lambdas,vector<int> &real_scal,vector<double> &scales)const;

	void computeChannels( const Mat &image,vector<Mat>& channels) const;

	void computeGradient(const Mat &img, 
                         Mat& grad1, 
                         Mat& grad2, 
                         Mat& qangle1,
                         Mat& qangle2,
                         Mat& mag_sum_s) const;	

	void setParas (const  detector_opt &in_para ) ;

	void compute_lambdas(const vector<Mat> &fold);

	bool convt_2_luv( const Mat input_image, 
					  Mat &L_channel,
					  Mat &U_channel,
					  Mat &v_channel) const;

    /* for color image this should be the first channel, (L for LUV, B for BGR), and the channels should be continuous in memory */
    bool computeGradMag( const Mat &input_image,      //in  : input image (first channel for color image ) 
                         const Mat &input_image2,     //in  : input image channel 2, set empty for gray image
                         const Mat &input_image3,     //in  : input image channel 3, set empty for gray image
                         Mat &mag,                    //out : output mag
                         Mat &ori,                    //out : output ori
                         bool full,                   //in  : ture -> 0-2pi, otherwise 0-pi
                         int channel = 0              //in  : choose specific channel to compute the mag and ori
                         ) const;

    bool conputeGradHist( const Mat &mag,             //in : mag
                          const Mat &ori,             //in : ori
                          Mat &Ghist,           //out: gradient hist 
                          int binSize,          //in : number of bin
                          int oritent,          //in : number of ori
                          bool full = false     //in : ture->0-2pi, false->0-pi
                          ) const;
	const detector_opt &getParas() const;

  private:

	  detector_opt m_opt;
	  	
	  vector<double>lam;

      Mat m_normPad;        //pad_size = normPad(5)
      Mat m_km;             //pad_size = smooth(1);

};
Mat get_Km(int smooth);
#endif

