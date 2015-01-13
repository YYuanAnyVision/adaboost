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
#include "Pyramid.h"


using namespace std;
using namespace cv;
void feature_Pyramids::convTri( const Mat &src, Mat &dst,Mat const &Km)
{
	
	filter2D(src,dst,src.depth(),Km,Point(-1,-1),0,IPL_BORDER_REFLECT);
	filter2D(dst,dst,src.depth(),Km.t(),Point(-1,-1),0,IPL_BORDER_REFLECT); 
	
}
void feature_Pyramids::computeGradient(const Mat &img, Mat& grad, Mat& qangle) 
{
	bool gammaCorrection = false;
    Size paddingTL=Size(0,0);
	Size paddingBR=Size(0,0);
	int nbins=m_opt.nbins;
	//CV_Assert( img.type() == CV_8U || img.type() == CV_8UC3 );
	CV_Assert( img.type() == CV_32F || img.type() == CV_32FC3 );

	Size gradsize(img.cols + paddingTL.width + paddingBR.width,
		img.rows + paddingTL.height + paddingBR.height);
	grad.create(gradsize, CV_32FC2);  // <magnitude*(1-alpha), magnitude*alpha>
	qangle.create(gradsize, CV_8UC2); // [0..nbins-1] - quantized gradient orientation
	Size wholeSize;
	Point roiofs;
	img.locateROI(wholeSize, roiofs);

	int x, y;
	int cn = img.channels();

	AutoBuffer<int> mapbuf(gradsize.width + gradsize.height + 4);
	int* xmap = (int*)mapbuf + 1;
	int* ymap = xmap + gradsize.width + 2;

	const int borderType = (int)BORDER_REFLECT_101;
	//! 1D interpolation function: returns coordinate of the "donor" pixel for the specified location p.
	for( x = -1; x < gradsize.width + 1; x++ )
		xmap[x] = borderInterpolate(x - paddingTL.width + roiofs.x,
		wholeSize.width, borderType) - roiofs.x;
	for( y = -1; y < gradsize.height + 1; y++ )
		ymap[y] = borderInterpolate(y - paddingTL.height + roiofs.y,
		wholeSize.height, borderType) - roiofs.y;

	// x- & y- derivatives for the whole row
	int width = gradsize.width;
	AutoBuffer<float> _dbuf(width*4);
	float* dbuf = _dbuf;
	Mat Dx(1, width, CV_32F, dbuf);
	Mat Dy(1, width, CV_32F, dbuf + width);
	Mat Mag(1, width, CV_32F, dbuf + width*2);
	Mat Angle(1, width, CV_32F, dbuf + width*3);

	int _nbins = nbins;
	float angleScale = (float)(_nbins/(2*CV_PI));
#ifdef HAVE_IPP
	Mat lutimg(img.rows,img.cols,CV_MAKETYPE(CV_32F,cn));
	Mat hidxs(1, width, CV_32F);
	Ipp32f* pHidxs  = (Ipp32f*)hidxs.data;
	Ipp32f* pAngles = (Ipp32f*)Angle.data;

	IppiSize roiSize;
	roiSize.width = img.cols;
	roiSize.height = img.rows;

	for( y = 0; y < roiSize.height; y++ )
	{
		const uchar* imgPtr = img.data + y*img.step;
		float* imglutPtr = (float*)(lutimg.data + y*lutimg.step);

		for( x = 0; x < roiSize.width*cn; x++ )
		{
			imglutPtr[x] = lut[imgPtr[x]];
		}
	}

#endif
	//vector<Mat> angle;
	//Mat combineMat;
	for( y = 0; y < gradsize.height; y++ )
	{
#ifdef HAVE_IPP
		const float* imgPtr  = (float*)(lutimg.data + lutimg.step*ymap[y]);
		const float* prevPtr = (float*)(lutimg.data + lutimg.step*ymap[y-1]);
		const float* nextPtr = (float*)(lutimg.data + lutimg.step*ymap[y+1]);
#else
		const float* imgPtr  = (float*)(img.data + img.step*ymap[y]);
		const float* prevPtr = (float*)(img.data + img.step*ymap[y-1]);
		const float* nextPtr = (float*)(img.data + img.step*ymap[y+1]);
#endif
		float* gradPtr = (float*)grad.ptr(y);
		uchar* qanglePtr = (uchar*)qangle.ptr(y);

		if( cn == 1 )
		{
			for( x = 0; x < width; x++ )
			{
				int x1 = xmap[x];
#ifdef HAVE_IPP
				dbuf[x] = (float)(imgPtr[xmap[x+1]] - imgPtr[xmap[x-1]]);
				dbuf[width + x] = (float)(nextPtr[x1] - prevPtr[x1]);
#else
				dbuf[x] = (float)(imgPtr[xmap[x+1]] - imgPtr[xmap[x-1]])*0.5f;
				dbuf[width + x] = (float)(nextPtr[x1] - prevPtr[x1])*0.5f; //??
#endif
			}
		}
		else
		{
			for( x = 0; x < width; x++ )
			{
				int x1 = xmap[x]*3;
				float dx0, dy0, dx, dy, mag0, mag;
#ifdef HAVE_IPP
				const float* p2 = imgPtr + xmap[x+1]*3;
				const float* p0 = imgPtr + xmap[x-1]*3;

				dx0 = p2[2] - p0[2];
				dy0 = nextPtr[x1+2] - prevPtr[x1+2];
				mag0 = dx0*dx0 + dy0*dy0;

				dx = p2[1] - p0[1];
				dy = nextPtr[x1+1] - prevPtr[x1+1];
				mag = dx*dx + dy*dy;

				if( mag0 < mag )
				{
					dx0 = dx;
					dy0 = dy;
					mag0 = mag;
				}

				dx = p2[0] - p0[0];
				dy = nextPtr[x1] - prevPtr[x1];
				mag = dx*dx + dy*dy;
#else
				const float* p2 = imgPtr + xmap[x+1]*3;
				const float* p0 = imgPtr + xmap[x-1]*3;

				dx0 = (p2[2] - p0[2])*0.5f;
				dy0 = (nextPtr[x1+2] - prevPtr[x1+2])*0.5f;
				mag0 = dx0*dx0 + dy0*dy0;

				dx = (p2[1] - p0[1])*0.5f;
				dy = (nextPtr[x1+1] - prevPtr[x1+1])*0.5f;
				mag = dx*dx + dy*dy;

				if( mag0 < mag )
				{
					dx0 = dx;
					dy0 = dy;
					mag0 = mag;
				}

				dx = (p2[0] - p0[0])*0.5f;
				dy = (nextPtr[x1] - prevPtr[x1])*0.5f;
				mag = dx*dx + dy*dy;
#endif
				if( mag0 < mag )
				{
					dx0 = dx;
					dy0 = dy;
					mag0 = mag;
				}

				dbuf[x] = dx0;
				dbuf[x+width] = dy0;
			}
		}
#ifdef HAVE_IPP
		ippsCartToPolar_32f((const Ipp32f*)Dx.data, (const Ipp32f*)Dy.data, (Ipp32f*)Mag.data, pAngles, width);
		for( x = 0; x < width; x++ )
		{
			if(pAngles[x] < 0.f)
				pAngles[x] += (Ipp32f)(CV_PI*2.);
		}

		ippsNormalize_32f(pAngles, pAngles, width, 0.5f/angleScale, 1.f/angleScale);
		ippsFloor_32f(pAngles,(Ipp32f*)hidxs.data,width);
		ippsSub_32f_I((Ipp32f*)hidxs.data,pAngles,width);
		ippsMul_32f_I((Ipp32f*)Mag.data,pAngles,width);

		ippsSub_32f_I(pAngles,(Ipp32f*)Mag.data,width);
		ippsRealToCplx_32f((Ipp32f*)Mag.data,pAngles,(Ipp32fc*)gradPtr,width);
#else
		cartToPolar( Dx, Dy, Mag, Angle, false );
#endif


		for( x = 0; x < width; x++ )
		{
#ifdef HAVE_IPP
			int hidx = (int)pHidxs[x];
#else
			float mag = dbuf[x+width*2], angle = dbuf[x+width*3]*angleScale - 0.5f;
			int hidx = cvFloor(angle);
			angle -= hidx;
			gradPtr[x*2] = mag*(1.f - angle);
			gradPtr[x*2+1] = mag*angle;
#endif
			if( hidx < 0 )
				hidx += _nbins;
			else if( hidx >= _nbins )
				hidx -= _nbins;
			assert( (unsigned)hidx < (unsigned)_nbins );

			qanglePtr[x*2] = (uchar)hidx;
			hidx++;
			hidx &= hidx < _nbins ? -1 : 0;
			qanglePtr[x*2+1] = (uchar)hidx;
		}
	}

}
void feature_Pyramids::computeChannels(const Mat &image,vector<Mat>& channels)
{

	/*set para*/
	int nbins=m_opt.nbins;
	int binsize=m_opt.binsize;
	/* compute luv and push */
	Mat_<float> grad;
	Mat_<float> angles;
	Mat gray,src,luv;
	int channels_addr_rows=(image.rows+binsize-1)/binsize;
	int channels_addr_cols=(image.cols+binsize-1)/binsize;
	Mat channels_addr=Mat::zeros((nbins+4)*channels_addr_rows,channels_addr_cols,CV_32FC1);
	if(image.channels() > 1)
	{
		src = Mat(image.rows, image.cols, CV_32FC3);
		image.convertTo(src, CV_32FC3, 1./255);
		cv::cvtColor(src, gray, CV_RGB2GRAY);
		cv::cvtColor(src, luv, CV_RGB2Luv);
	}else{
		src = Mat(image.rows, image.cols, CV_32FC1);
		image.convertTo(src, CV_32FC1, 1./255);
		src.copyTo(gray);
	}
	channels.clear();
	if(image.channels() > 1)
	{
		Mat luv_channels[3];
		cv::split(luv, luv_channels);
	
		/*  0<L<100, -134<u<220 -140<v<122  */
		/*  normalize to [0, 1] */
		luv_channels[0] *= 1.0/354;
		luv_channels[1] = (luv_channels[1]+134)/(354.0);
		luv_channels[2] = (luv_channels[2]+140)/(354.0);
		for( int i = 0; i < 3; ++i )
		{
			Mat channels_tmp=channels_addr.rowRange(i*channels_addr_rows,(i+1)*channels_addr_rows);
		    resize(luv_channels[i],channels_tmp,channels_tmp.size(),0.0,0.0,1);
			channels.push_back(channels_tmp);
		}
	}

	/*compute gradient*/
	Mat mag,ori;
	Mat mag_tmp;
	Mat mag_sum=channels_addr.rowRange(3*channels_addr_rows,4*channels_addr_rows);
	computeGradient(src, mag, ori);
	resize(mag,mag_tmp,mag_sum.size(),0.0,0.0,1);
	vector<Mat> mag_split;
	cv::split(mag_tmp, mag_split);
	mag_sum=mag_split[0]+mag_split[1];
	channels.push_back(mag_sum);

	/*compute grad_hist*/
	vector<Mat> bins_mat;
	int bins_mat_rows=(mag.rows+binsize-1)/binsize;
	int bins_mat_cols=(mag.cols+binsize-1)/binsize;
	for( int s=0;s<nbins;s++){
		Mat channels_tmp=channels_addr.rowRange((s+4)*channels_addr_rows,(s+5)*channels_addr_rows);
		bins_mat.push_back(channels_tmp);}
	/*split*/
#define GH \
	bins_mat[ori.at<Vec2b>(row,col)[0]].at<float>((row)/binsize,(col)/binsize)+=mag.at<Vec2f>(row,col)[0];\
	bins_mat[ori.at<Vec2b>(row,col)[1]].at<float>((row)/binsize,(col)/binsize)+=mag.at<Vec2f>(row,col)[1];
	for(int row=0;row<mag.rows;row++){
		for(int col=0;col<mag.cols;col++){GH;}}
	/*push*/
	for (int c=0;c < (int)bins_mat.size();c++){

		channels.push_back(bins_mat[c]);
	}
	/*  check the pointer */
	/*float *add1 = (float*)channels[0].data;
	for( int c=1;c<channels.size();c++)
	{
		cout<<"pointer "<<(static_cast<void*>(channels[c].data) == (static_cast<void*>(add1+c*channels_addr_rows*channels_addr_cols))? true :false)<<endl;
	}*/
}
void feature_Pyramids:: chnsPyramid(const Mat &img,  vector<vector<Mat> > &approxPyramid)
{
	int nPerOct = m_opt.nPerOct;
	int nOctUp  = m_opt.nOctUp;
	int shrink  = m_opt.shrink;
	int smooth  = m_opt.smooth;
	int nbins   = m_opt.nbins;
	int binsize = m_opt.binsize;
	int nApprox = m_opt.nApprox;
	Size diam   = m_opt.diam;
	/*get scales*/
	float minDs=(float)min(diam.width,diam.height);
	int nscales=(int)floor(nPerOct*(nOctUp+log(min(img.cols/minDs,img.rows/minDs))/log(2))+1);
	vector<float> scales;
	vector<Size> ap_size;
	Size ap_tmp_size;

	CV_Assert(nApprox<nscales);
	/*compute real & approx scales*/
	vector<int> real_scal;
	float d0=(float)min(img.rows,img.cols);
	float d1=(float)max(img.rows,img.cols);
	for (float s=0;s<nscales;s++)
	{
		if (((int)s%(nApprox+1)==0))
		{
			real_scal.push_back((int)s);
		}
		if ((s==(nscales-1)&&(s>real_scal[real_scal.size()-1])&&(s-real_scal[real_scal.size()-1]>(nApprox+1)/2)))
		{
			real_scal.push_back((int)s);
		}
		/*adjust ap_size*/
		double sc=pow(2.0,(-s)/nPerOct+nOctUp);
		double s0=(cvRound(d0*sc/shrink)*shrink-0.25*shrink)/d0;
		double s1=(cvRound(d0*sc/shrink)*shrink+0.25*shrink)/d0;
		float ss,es1,es2,a=10,val;
		for(int c=0;c<101;c++)
		{
			ss=(float)((s1-s0)*c/101+s0);
			es1=abs(d0*ss-cvRound(d0*ss/shrink)*shrink);
			es2=abs(d1*ss-cvRound(d1*ss/shrink)*shrink);
			if (max(es1,es2)<a)
			{
				a=max(es1,es2);
				val=ss;
			}
		}
		scales.push_back(val);
		/*save ap_size*///error
		ap_tmp_size.height=cvRound(((img.rows+binsize-1)/binsize)*val/shrink);
		ap_tmp_size.width=cvRound(((img.cols+binsize-1)/binsize)*val/shrink);
		ap_size.push_back(ap_tmp_size);
	}
	//compute real 
	Mat img_tmp;
	vector<vector<Mat> > chns_Pyramid;
	int chns_num;
	for (int s_r=0;s_r<(int)real_scal.size();s_r++)
	{
		vector<Mat> chns;
		resize(img,img_tmp,ap_size[real_scal[s_r]],0.0,0.0,1);
		computeChannels(img_tmp,chns);
		//test
		chns_num=chns.size();
		chns_Pyramid.push_back(chns);
	}
	//compute lambdas
	vector<double> lambdas;
	Scalar lam_s;
	Scalar lam_ss;
	CV_Assert(chns_Pyramid.size()>=2);
	if (chns_Pyramid.size()>2)
	{
		for (int i = 0; i < chns_num; i++)
		{
			lam_s=sum(chns_Pyramid[1][i])/(chns_Pyramid[1][i].rows*chns_Pyramid[1][i].cols*1.0);
			lam_ss=sum(chns_Pyramid[2][i])/(chns_Pyramid[2][i].rows*chns_Pyramid[2][i].cols*1.0);
			//compute lambdas
			lambdas.push_back(-cv::log(lam_ss.val[0]/lam_s.val[0])/cv::log(scales[real_scal[2]]/scales[real_scal[1]]));
		}
	}else{
		for (int i = 0; i < chns_num; i++)
		{
			lam_s=sum(chns_Pyramid[0][i])/(chns_Pyramid[0][i].rows*chns_Pyramid[0][i].cols*1.0);
			lam_ss=sum(chns_Pyramid[1][i])/(chns_Pyramid[1][i].rows*chns_Pyramid[1][i].cols*1.0);
			//compute lambdas
			lambdas.push_back(-cv::log(lam_ss.val[0]/lam_s.val[0])/cv::log(scales[real_scal[1]]/scales[real_scal[0]]));
		}
	}
	//compute approx ÌØÕ÷»ù×¼
	vector<int> approx_scal;
	for (int s_r=0;s_r<nscales;s_r++)
	{
		int tmp=s_r/(nApprox+1);
		if (s_r-real_scal[tmp]>((nApprox+1)/2))
		{
			approx_scal.push_back(real_scal[tmp+1]);
		}else{
			approx_scal.push_back(real_scal[tmp]);
		}	
	}
	//compute the filter
	Mat Km;
	float *kern=new float[2*smooth+1];
	for (int cout=0;cout<=smooth;cout++)
	{
		kern[cout]=(float)((cout+1)/((smooth+1.0)*(smooth+1.0)));
		kern[2*smooth-cout]=kern[cout];
	}
	Km=Mat(1,(2*smooth+1),CV_32FC1,kern); 
	//compute approxPyramid
	float ratio;
	for (int ap_id=0;ap_id<(int)approx_scal.size();ap_id++)
	{
		vector<Mat> approx_chns;
		approx_chns.clear();

		for(int n_chans=0;n_chans<chns_num;n_chans++)
		{
			Mat py;
			int ma=approx_scal[ap_id]/(nApprox+1);
			resize(chns_Pyramid[ma][n_chans],py,ap_size[ap_id],0.0,0.0,1);
			ratio=(float)pow(scales[ap_id]/scales[approx_scal[ap_id]],-lambdas[n_chans]);
			py=py*ratio;
			//smooth channels, optionally pad and concatenate channels
			convTri(py,py,Km);
			approx_chns.push_back(py);
		} 
		approxPyramid.push_back(approx_chns);
	}
	/*delete*/
	delete kern;
    vector<int>().swap(real_scal);
	vector<int>().swap(approx_scal);
	vector<double>().swap(lambdas);
	vector<vector<Mat> >().swap(chns_Pyramid);
}
void feature_Pyramids::setParas(const detector_opt &in_para)
{
     m_opt = in_para;
}
const detector_opt& feature_Pyramids::getParas() const
{
	 return m_opt;
}
feature_Pyramids::feature_Pyramids()
{	
}
feature_Pyramids::~feature_Pyramids()
{
}

