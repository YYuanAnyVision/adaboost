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

	void convTri( const Mat &src, Mat &dst,const Mat &Km) const;        
  
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


	const detector_opt &getParas() const;




    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  computeChannels_sse
     *  Description:  compute channel features, same effect as computeChannels, only use sse
     *                channels data is continuous in momory, LUVGOG1G2G3G4G5G6 
     * =====================================================================================
     */
    bool computeChannels_sse( const Mat &image,             // in : input image, BGR 
                              vector<Mat>& channels) const; //out : 10 channle features, continuous in memory
    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  convTri
     *  Description:   convolve one row of I by a 2rx1 triangle filter
     * =====================================================================================
     */
    // sse version, faster
	void convTri( const Mat &src,       // in : input data, for color image, this is the first channel, and set dim =3
                  Mat &dst,             // out: output data, for color image, this is the first channel, and set dim =3
                  int conv_size,        // in : value of r, the length of the kernel
                  int dim) const;       // in : dim, DO NOT SET dim=3 for gray image, and should make 3 channels continuous 


    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  convt_2_luv
     *  Description:  convert the image from BGR to LUV, LUV channel is contiunous in memory
     * =====================================================================================
     */

	bool convt_2_luv( const Mat input_image,            // in : input image
					  Mat &L_channel,                   // out: L channel
					  Mat &U_channel,                   // out: U channel
					  Mat &v_channel) const;            // out: V channel 


    /* 
     * ===  FUNCTION  ======================================================================
     *         Name:  computeGradMag
     *  Description:  compute gradient magnitude and orientation at each location (uses sse)\
     *                for color image this should be the first channel, (L for LUV, B for BGR), and the channels should be continuous in memory 
     * =====================================================================================
     */
     bool computeGradMag( const Mat &input_image,      //in  : input image (first channel for color image ) 
                          const Mat &input_image2,     //in  : input image channel 2, set empty for gray image
                          const Mat &input_image3,     //in  : input image channel 3, set empty for gray image
                          Mat &mag,                    //out : output mag
                          Mat &ori,                    //out : output ori
                          bool full,                   //in  : ture -> 0-2pi, otherwise 0-pi
                          int channel = 0              //in  : choose specific channel to compute the mag and ori
                       ) const;


     /* 
      * ===  FUNCTION  ======================================================================
      *         Name:  computeGradHist
      *  Description:  compute the Gradient Hist for oritent, output size will be  w/binSize*oritent x h/binSize, alse continuous in momory
      * =====================================================================================
      */
      bool computeGradHist( const Mat &mag,           //in : mag -> size w x h
                            const Mat &ori,           //in : ori -> same size with mag
                            Mat &Ghist,               //out: gradient hist size - > w/binSize*oritent x h/binSize
                            int binSize,              //in : size of bin, degree of aggregatation
                            int oritent,              //in : number of orientations, eg 6;
                            bool full = false         //in : ture->0-2pi, false->0-pi
                            ) const;


      /* 
       * ===  FUNCTION  ======================================================================
       *         Name:  chnsPyramid_sse
       *  Description:  compute channels pyramid without approximation, slower but accurate
       * =====================================================================================
       */
    bool chnsPyramid_sse(const Mat &img,                        //in : input image
                         vector<vector<Mat> > &chns_Pyramid,    //out: output features
                         vector<double> &scales) const;         //out: scale of each pyramid


  private:

	  detector_opt m_opt;
	  	
	  vector<double>lam;

      Mat m_normPad;        //pad_size = normPad(5)
      Mat m_km;             //pad_size = smooth(1);

};
Mat get_Km(int smooth);
#endif

