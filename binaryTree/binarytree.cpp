#include <iostream>
#include <assert.h>
#include <vector>
#include <cmath>
#include <omp.h>

#include <algorithm>    // std::min
#include "opencv2/highgui/highgui.hpp"
#include "binarytree.hpp"

using namespace cv;
using namespace std;

bool binaryTree::checkTreeParas( const tree_para & p )		/* input parameter */
{
	if( p.nBins > 256||p.nBins < 2 || p.maxDepth< 0 || p.minWeight < 0 || p.minWeight > 1 || p.nThreads < 0 || 
			p.fracFtrs >1 || p.fracFtrs<0)
		return false;
	else
		return true;
}

void binaryTree::computeXMinMax( const Mat &X0,		/* neg data */
								 const Mat &X1,		/* pos data , column feature */
								 Mat& XMin,			
								 Mat& XMax)
{
	assert( X0.rows == X1.rows );
	int featureDim = X0.rows;
	
	for ( int i=0;i<featureDim ;i++ ) 
	{
		double max1, max2, min1, min2;
		minMaxLoc( X0.row(i), &min1, &max1);
		minMaxLoc( X1.row(i), &min2, &max2);
		XMin.at<double>(i,0) = std::min( min1, min2 ) - 0.01;
		XMax.at<double>(i,0) = std::max( max1, max2 ) + 0.01;
	}
}


bool binaryTree::binaryTreeTrain(   
						const Mat &neg_data,			// in column feature                    uint8   featuredim X number
						const Mat &pos_data,			// in same as neg_data			        uint8	featuredim X number
						const Mat &norm_neg_weight,		// in sample weight                    double   featuredim X 1 
						const Mat &norm_pos_weight,		// in same as above					   double	featuredim X 1
						int nBins,						// in number of bins
						double prior,					// in prior of the error rate
						const Mat &fids_st,				// in index of the selected feature				featuredim X 1
						int nthreads,					// in numbers of the threads use in training
						Mat &errors_st,					// out
						Mat &thresholds)				// out
{
	if( neg_data.type()!= CV_8U || pos_data.type() != CV_8U ||
			norm_neg_weight.type()!= CV_64F || norm_pos_weight.type()!= CV_64F ||
			fids_st.type() != CV_32S || nthreads < 0 || nBins < 0 || prior < 0)
	{
		cout<<"wrong input ..."<<endl;
		return false;
	}

	int number_of_selected_feature = fids_st.rows;
	errors_st = Mat::zeros( number_of_selected_feature, 1, CV_64F );
	thresholds = Mat::zeros( number_of_selected_feature, 1, CV_64F);

	int Nthreads = std::min( nthreads, omp_get_max_threads());
	cout<<"training using "<<Nthreads<<" threads ..."<<endl;

	#pragma omp parallel for num_threads(nThreads)
	for ( int i=0;i<number_of_selected_feature ;i++ ) 
	{
		
	}

}


bool binaryTree::Train( const Mat &neg_data,			/* input, format-> featuredim x number0 */
						const Mat &pos_data,			/* input, format-> featuredim x number1 */ 
						const tree_para &paras			/* input tree paras */)
{
	/* sanity check*/
	if( neg_data.empty() || pos_data.empty() || !checkTreeParas(paras) || neg_data.channels()!=1 || pos_data.channels()!=1)
	{
		cout<<"input wrong format"<<endl;
		return false;
	}

	if( neg_data.type() != pos_data.type() )
	{
		cout<<"neg and pos data should be the same type "<<endl;
		return false;
	}

	int feature_dim = neg_data.rows;
	if( feature_dim != pos_data.rows)
	{
		cout<<"feature dim should be the same between neg and pos samples "<<endl;
		return false;
	}

	int num_neg_samples = neg_data.cols;
	int num_pos_samples = pos_data.cols;
	
	Mat Xmin( feature_dim , 1, CV_64F );
	Mat Xmax( feature_dim , 1, CV_64F );
	Mat Xstep( feature_dim , 1, CV_64F );

	/*  compute min and max value , used for value quantization*/
	computeXMinMax( neg_data, pos_data, Xmin, Xmax);

	/*  0 for neg, 1 for pos */
	/*  wts0 = 1/num_neg_samples, wts0=wts0/sum(wts0) + sum(wts1)  and 2 = sum(wts0) + sum(wts1)*/
	Mat wts0 = Mat::ones( num_neg_samples, 1, CV_64F); wts0 /=(num_neg_samples*2);
	Mat wts1 = Mat::ones( num_pos_samples, 1, CV_64F); wts1 /=(num_pos_samples*2);
	
	Mat quan_neg_data( neg_data.size(), CV_64F );
	Mat quan_pos_data( pos_data.size(), CV_64F );

	if( neg_data.type() == CV_8U)
	{
		quan_neg_data = neg_data;
		quan_pos_data = pos_data;
	} 
	else
	{
		// data quantization, range [0 paras.nbins]
		Xstep = ( Xmax - Xmin )/( paras.nBins - 1);
		for ( int i=0;i<neg_data.cols ;i++ ) 
		{
			Mat tmp = (neg_data.col(i) - Xmin)/Xstep; /*  Mat expression used in assignment, no mem copy here */
			tmp.copyTo( quan_neg_data.col(i));		
		}
		
		for ( int i=0;i<pos_data.cols ;i++ ) 
		{
			Mat tmp = (pos_data.col(i) - Xmin)/Xstep;
			tmp.copyTo( quan_pos_data.col(i));
		}
		/*  convert to uint8  */
		quan_pos_data.convertTo( quan_pos_data, CV_8U);
		quan_neg_data.convertTo( quan_neg_data, CV_8U);
	}
	
	//cout<<"quantization data "<<quan_neg_data.col(308)<<endl;
	/*  K--> max number of split */
	int K = 2*( num_neg_samples + num_pos_samples);
	m_tree.thrs    = Mat::zeros( K, 1, neg_data.type());
	m_tree.hs      = Mat::zeros( K, 1, CV_64F);
	m_tree.weights = Mat::zeros( K, 1, CV_64F);
	m_tree.fids =  Mat::zeros( K, 1, CV_32S);
	m_tree.child = Mat::zeros( K, 1, CV_32S);
	m_tree.depth = Mat::zeros( K, 1, CV_32S);
	Mat errs = Mat( K, 1, CV_64F);

	/* store the weight of all nodes, initialize weight in the root node( index 0 )
	 * delete corresponding item after the node split 
	 * the tree splits in a breath-first manner */
	vector<Mat*> wtsAll0;wtsAll0.reserve(K);wtsAll0[0] = &wts0;
	vector<Mat*> wtsAll1;wtsAll1.reserve(K);wtsAll1[0] = &wts1;
	
	int k=0;    /* k is the index now processing ... */
	K=1;		/* increasing in the training process */

	/*  ready to go , train decission tree classifier*/
	Mat fidsSt( feature_dim, 1, CV_32S); /* pre computes the feature indexs, sampled later for feature selection*/
	for ( int i=0;i<feature_dim ;i++) 
		fidsSt.at<int>(i,0) = i;
	cv::RNG rng(getTickCount());

	while( k < K)
	{
		/* get node wrights and prior */
		Mat *weight0 = wtsAll0[k]; wtsAll0[k] = NULL;
		Scalar tmp_sum=cv::sum(*weight0); double w0=tmp_sum.val[0]; /* delete weight0 later */
		Mat *weight1 = wtsAll1[k]; wtsAll1[k]= NULL;
		tmp_sum = cv::sum(*weight1); double w1 = tmp_sum.val[0];
		
		double w = w0+w1; double prior = w1/w; 
		m_tree.weights.at<double>(k,0) = w; errs.at<double>(k,0)=std::min( prior, 1-prior);
		m_tree.hs.at<double>(k,0)=std::max( -4.0, std::min(4.0, 0.5*std::log(prior/(1-prior))));

		//cout<<"prior is "<<prior<<endl;
		//cout<<m_tree.weights.at<double>(k,0)<<endl;
		//cout<<m_tree.hs.at<double>(k,0)<<endl;
		
		/*  if nearly pure node ot insufficient data --> don't train split */
		if( prior < 1e-3 || prior > 1-1e-3 || m_tree.depth.at<unsigned int>(k,0) > paras.maxDepth)
		{
			cout<<"------- node number "<<k<<" stop spliting -------"<<endl;
			k++;continue;	/*  not break, since there maybe other node needs to split */
		}

		/*  randomly select feature index  */
		cv::randShuffle( fidsSt, 1, &rng);
		
		/*  ---------------------- train ----------------------- */
		Mat errors_st, threshold_st;
		binaryTreeTrain( quan_neg_data, quan_pos_data, *(weight0)/w, *(weight1)/w,  paras.nBins, prior, 
				 fidsSt.rowRange(0,int(fidsSt.rows*paras.fracFtrs)), paras.nThreads, errors_st, threshold_st);
		
		k++;
	}


	return true;
}











