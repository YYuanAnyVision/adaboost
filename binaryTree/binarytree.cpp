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

template<class T> bool _any( T *ptr, int numberOfele )
{
	
	for ( int c=0;c<numberOfele ;c++ )
	{
		if( 1.0-ptr[c] > 1e-7 || ptr[c] - 1.0 > 1e-7)
			return true;
	}
	return false;
}

bool binaryTree::checkTreeParas( const tree_para & p )		const  /* input parameter */
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
								 Mat& XMax ) const
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


void binaryTree::convertHsToDouble()
{
	if( !m_tree.hs.empty() )
	{
		for( int c=0;c<m_tree.hs.rows;c++)
			m_tree.hs.at<double>(c,0) = ( m_tree.hs.at<double>(c,0) > 1e-7?1:-1);
	}
}

bool binaryTree::computeCDF(	const Mat & sampleData,				// in samples	1 x numberOfSamples, same feature for all the samples, one row
								const Mat & weights,				// in weights	numberOfSamples x 1
								int nBins,							// in number of bins
								vector<double> &cdf					// out cdf
								) const
{
	/*  initialize the cdf */
	for ( int c=0;c<cdf.size() ;c++ )
		cdf[c] = 0.0;
	if( sampleData.type() != CV_8U || weights.type() != CV_64F || nBins != cdf.size() || sampleData.rows != 1 || sampleData.cols != weights.rows)
	{
		cout<<"in function computeCDF : Wrong Data Formar in function computeCDF, return "<<endl;
		return false;
	}

	int number_of_samples = sampleData.cols;

	uchar *p = sampleData.data;										/* just one row ~, same as uchar *p = sampleData.ptr(0) */
	double *w = (double*)(weights.data);							/*  just one col, also continus */

	/* for each sample, computing its bin*/
	for ( int c=0;c<number_of_samples ;c++ ) 
	{
		cdf[ p[c] ] += w[c];
	}

	/* culmulating ...*/
	for( int c=1;c<cdf.size();c++)
	{
		cdf[c] += cdf[c-1];	
	}

}

bool binaryTree::binaryTreeTrain(   
						const Mat &neg_data,			// in column feature                    uint8   featuredim X number neg
						const Mat &pos_data,			// in same as neg_data			        uint8	featuredim X number pos
						const Mat &norm_neg_weight,		// in sample weight                    double   number neg X 1 
						const Mat &norm_pos_weight,		// in same as above					   double	number pos X 1
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
		cout<<"in functin binaryTreeTrain : wrong input ..."<<endl;
		return false;
	}

	int number_of_selected_feature = fids_st.rows;
	errors_st = Mat::zeros( number_of_selected_feature, 1, CV_64F );
	thresholds = Mat::zeros( number_of_selected_feature, 1, CV_8U);

	int Nthreads = std::min( nthreads, omp_get_max_threads());

	/* debug information:  */
	//cout<<"number of selected feature is "<<fids_st.rows<<"\n: "<<fids_st<<endl;


	/*  choose the best feature with the best threshold, so iteration is between number_of_selected_feature*nBins  */
	#pragma omp parallel for num_threads(Nthreads)
	for ( int i=0;i<number_of_selected_feature ;i++ ) 
	{
		vector<double> cdf0(nBins, 0);
		vector<double> cdf1(nBins, 0);

		double e0 = 1;		/* e0 and e1 --> e0+e1 = 1, getting a very small e0 and very big e0 are both good*/
		double e1 = 0;
		double e;
		int thr;			/*  threshold, now is between [1, nBins-1] integer*/
		computeCDF( neg_data.row(fids_st.at<int>(i,0)), norm_neg_weight, nBins, cdf0 );
		computeCDF( pos_data.row(fids_st.at<int>(i,0)), norm_pos_weight, nBins, cdf1 );
		for ( int c=0;c<nBins ; c++) 
		{
			 e = prior -cdf1[c] + cdf0[c];
			 if(e<e0)
			 {
				 e0 = e;e1 = 1-e;thr=c;
			 }
			 else if(e>e1)
			 {
				 e0 = 1-e;e1 = e;thr=c;
			 }

		}
		errors_st.at<double>(i,0) = e0;
		thresholds.at<uchar>(i,0) = thr;
	}
	return true;
}

bool binaryTree::any( const Mat &input ) const
{
	if( input.channels()!=1 || !input.isContinuous())
		return false;
	int nrows = input.rows;
	int ncols = input.cols;
	if( input.type() == CV_8U)
		return _any((uchar*)input.data, nrows*ncols);
	else if( input.type() == CV_32F)
		return _any( (float*)input.data, nrows*ncols);
	else if( input.type() == CV_64F)
		return _any( (double*)input.data, nrows*ncols);
	else if( input.type() == CV_32S)
		return _any( (int*)input.data, nrows*ncols);
	else
		return false;
}

void binaryTree::SetDebug( bool isDebug )		/* in: wanna debug information */
{
	m_debug = isDebug;
}

bool binaryTree::Train( data_pack & train_data,			/* input&output : training data and weights info */
						const tree_para &paras)			/* input tree paras */
{
	if(m_debug)
	{
		cout<<"remember the input data will be revised, make a copy before Train function !"<<endl;
		cout<<"training parameters are: \n";
		cout<<"nbins(shoule be less than 256:) :\t\t"<<paras.nBins<<endl;
		cout<<"maxDepth:\t\t"<<paras.maxDepth<<endl;
		cout<<"fracFtrs:\t\t"<<paras.fracFtrs<<endl;
		cout<<"nThreads:\t\t"<<paras.nThreads<<endl;
	}
	/*  extract data from paclage .. */
	Mat neg_data = train_data.neg_data;
	Mat pos_data = train_data.pos_data;

	/* sanity check*/
	if( neg_data.empty() || pos_data.empty() || !checkTreeParas(paras) || neg_data.channels()!=1 || pos_data.channels()!=1)
	{
		cout<<"in function Train : input wrong format"<<endl;
		return false;
	}

	if( neg_data.type() != pos_data.type() )
	{
		cout<<"in function Train : neg and pos data should be the same type "<<endl;
		return false;
	}

	int feature_dim = neg_data.rows;
	if( feature_dim != pos_data.rows)
	{
		cout<<"in function Train : feature dim should be the same between neg and pos samples "<<endl;
		return false;
	}

	int num_neg_samples = neg_data.cols;
	int num_pos_samples = pos_data.cols;
	
	/*  data is not quantized at the first time, since quantization is very expensive, keep
	 *  the information */
	Mat Xmin;	
	Mat Xmax;	
	Mat Xstep;	
	if( train_data.Xmax.empty() || train_data.Xmin.empty() || train_data.Xstep.empty())
	{
		if(m_debug)
			cout<<"quantization info is empty, generate new quantization infos "<<endl;
		Xmin	= Mat::zeros( feature_dim , 1, CV_64F );
		Xmax	= Mat::zeros( feature_dim , 1, CV_64F );
		Xstep	= Mat::zeros( feature_dim , 1, CV_64F );

		/*  compute min and max value , used for value quantization*/
		computeXMinMax( neg_data, pos_data, Xmin, Xmax);
		Xstep = ( Xmax - Xmin )/( paras.nBins - 1);

		/*  keep quantization information, quantization info is read only, no need to copy*/
		train_data.Xmax  = Xmax;
		train_data.Xmin  = Xmin;
		train_data.Xstep = Xstep;
	}
	else
	{	
		if(m_debug)
			cout<<"extracting quantization info from the train_data directly "<<endl;
		Xmin  = train_data.Xmin;
	 	Xmax  = train_data.Xmax;
		Xstep = train_data.Xstep; 
	}

	/* extract weights informatin */
	Mat *wts0,*wts1;
	if( train_data.wts0.empty())
	{
		if(m_debug)
			cout<<"empty weight, generate new uniform wright "<<endl;
		wts0 = new Mat(Mat::ones( num_neg_samples, 1, CV_64F)); *wts0 /=(num_neg_samples);

		/*  save the weight, clone the Mat since wts0 wts1 will be delete in the loop */
		train_data.wts0 = (*wts0).clone();
	}
	else
	{
		if(m_debug)
			cout<<"extracting weights0  info from the train_data directly "<<endl;
		wts0 = new Mat(train_data.wts0);
	}

	if( train_data.wts1.empty())
	{
		if(m_debug)
			cout<<"empty weight, generate new uniform wright "<<endl;
		wts1 = new Mat(Mat::ones( num_pos_samples, 1, CV_64F)); *wts1 /=(num_pos_samples);
		/*  save the weight, clone the Mat since wts0 wts1 will be delete in the loop */
		train_data.wts1 = (*wts1).clone();
	}
	else
	{
		if(m_debug)
			cout<<"extracting weights1 info from the train_data directly "<<endl;
		wts1 = new Mat(train_data.wts1);
	}

	/*  normalize the weights if necessary*/
	double w = cv::sum(*wts0)[0] + cv::sum(*wts1)[0];
	if( m_debug)
	{
		cout<<"w0 is now \t"<<cv::sum(*wts0)[0]<<endl;
		cout<<"w1 is now \t"<<cv::sum(*wts1)[0]<<endl;
		cout<<"w is now \t"<<w<<endl;
	}
	if( std::abs( w - 1.0) > 1e-3)
	{
		*wts0 = *wts0/w;
		*wts1 = *wts1/w;
	}
	
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

		/* ------- save the quantized data to train_data pack------  
		 * !!!! this will change the original data !!!! */
		train_data.neg_data = quan_neg_data;
		train_data.pos_data = quan_pos_data;
	}
	
	//cout<<"quantization data "<<quan_neg_data.col(308)<<endl;
	/*  K--> max number of split */
	int K = 2*( num_neg_samples + num_pos_samples);
	m_tree.thrs    = Mat::zeros( K, 1, CV_64F);
	m_tree.hs      = Mat::zeros( K, 1, CV_64F);
	m_tree.weights = Mat::zeros( K, 1, CV_64F);
	m_tree.fids =  Mat::zeros( K, 1, CV_32S);
	m_tree.child = Mat::zeros( K, 1, CV_32S);
	m_tree.depth = Mat::zeros( K, 1, CV_32S);
	Mat errs = Mat( K, 1, CV_64F);

	/* store the weight of all nodes, initialize weight in the root node( index 0 )
	 * delete corresponding item after the node split 
	 * the tree splits in a breath-first manner */
	vector<Mat*> wtsAll0;wtsAll0.reserve(K);wtsAll0[0] = wts0;
	vector<Mat*> wtsAll1;wtsAll1.reserve(K);wtsAll1[0] = wts1;
	
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
		Mat *weight0 = wtsAll0[k]; 
		Scalar tmp_sum=cv::sum(*weight0); double w0=tmp_sum.val[0]; /* delete weight0 later */
		Mat *weight1 = wtsAll1[k];
		tmp_sum = cv::sum(*weight1); double w1 = tmp_sum.val[0];
		
		double w = w0+w1; double prior = w1/w; 
		m_tree.weights.at<double>(k,0) = w; errs.at<double>(k,0)=std::min( prior, 1-prior);
		m_tree.hs.at<double>(k,0)=std::max( -4.0, std::min(4.0, 0.5*std::log(prior/(1-prior))));

		//cout<<"prior is "<<prior<<endl;
		//cout<<m_tree.weights.at<double>(k,0)<<endl;
		//cout<<m_tree.hs.at<double>(k,0)<<endl;
		
		/*  if nearly pure node ot insufficient data --> don't train split */
		if( prior < 1e-3 || prior > 1-1e-3 || m_tree.depth.at<int>(k,0) >= paras.maxDepth || w < paras.minWeight)
		{
			if(m_debug)
				cout<<"------- node number "<<k<<" stop spliting -------"<<endl;
			k++;continue;	/*  not break, since there maybe other node needs to split */
		}

		/*  randomly select feature index  */
		cv::randShuffle( fidsSt, 1, &rng);
		
		/*  ---------------------- train ----------------------- */
		Mat errors_st, threshold_st;
		binaryTreeTrain( quan_neg_data, quan_pos_data, *(weight0)/w, *(weight1)/w,  paras.nBins, prior, 
				 fidsSt.rowRange(0,int(fidsSt.rows*paras.fracFtrs)), paras.nThreads, errors_st, threshold_st);
		
		/* find the minimum error, and corresponding feature index */
		Point minLocation; double minError; double maxError;
		cv::minMaxLoc( errors_st, &minError, &maxError, &minLocation);
		int minErrorIndex = (int)minLocation.y;
		//cout<<"min error is "<<minError<<" max error is "<<maxError<<" index: "<<minLocation.y<<endl;

		int selectedFeature = fidsSt.at<int>(minErrorIndex, 0);
		//cout<<"selected feature is No "<<selectedFeature<<endl;

		double threshold_ready_to_apply = threshold_st.at<uchar>(minErrorIndex,0) + 0.5;
		//cout<<"threshold is set to "<<threshold_ready_to_apply<<endl;

		/* split the data and continue if necessary */
		Mat left0 = quan_neg_data.row(selectedFeature) < threshold_ready_to_apply; left0 /=255; /* normalize to 0,1 otherwize is 0,255 */
		Mat left1 = quan_pos_data.row(selectedFeature) < threshold_ready_to_apply; left1 /=255;
		

		/* any(left0) --> there neg samples left in the left leaf, so the condition means they will split, 
		 *  !any(left0) && !any(left1) --> means all the sample in right node ..*/
		if( (any(left0) || any(left1)) && ( any(1-left0) || any(1-left1)))
		{
			if(m_debug)
				cout<<"--------> split <----------"<<endl;
			double actual_threshold = Xmin.at<double>( selectedFeature, 0) + Xstep.at<double>( selectedFeature, 0) * threshold_ready_to_apply;
			m_tree.child.at<int>(k,0) = K; m_tree.fids.at<int>(k,0) = selectedFeature;m_tree.thrs.at<double>(k,0) = actual_threshold;
			if(m_debug)
				cout<<"---> split node "<<k<<"'s child is "<<m_tree.child.at<int>(k,0)<<", with feature "<<
				m_tree.fids.at<int>(k,0)<<" and threshold "<<m_tree.thrs.at<double>(k,0)<<" with depth "<<(int)m_tree.depth.at<int>(k,0)<<endl;

			/* -------------------------------- weights rearrange -------------------------------------*/
			Mat left0_double; left0.convertTo( left0_double, CV_64F);
			//cout<<"weight0 size "<<(*weight0).rows<<" "<<(*weight0).cols<<" "<<(*weight0).type()<<endl;
			//cout<<"left0 size "<<left0_double.rows<<" "<<left0_double.cols<<" "<<left0_double.type()<<endl;
			Mat *newWeight0 = new Mat((*weight0).mul(left0_double.t()));
			Mat *newWeight0plus = new Mat((*weight0).mul( 1-left0_double.t())); /* "1-left0_double" is same as "~left0_double" */
			//cout<<" weights0 is \n"<<*weight0<<endl;
			//cout<<"left0_double is \n"<<left0_double<<endl;
			//cout<<" newWeights is \n"<<newWeight0<<endl;
			wtsAll0[K]   = newWeight0;				/* left node */
			wtsAll0[K+1] = newWeight0plus;			/* right node, index+1*/
			delete wtsAll0[k]; wtsAll0[k] = NULL;	/* works on node k is done, release the memory */

			Mat left1_double; left1.convertTo( left1_double, CV_64F);
			Mat *newWeight1 = new Mat( (*weight1).mul(left1_double.t()) );
			Mat *newWeight1plus = new Mat( (*weight1).mul(1-left1_double.t()) );

			wtsAll1[K] = newWeight1;
			wtsAll1[K+1] = newWeight1plus; 
			delete wtsAll1[k]; wtsAll1[k] = NULL;
			
			/* -------------------------------- depth increasing-------------------------------------*/
			m_tree.depth.at<uchar>(K,0) = m_tree.depth.at<int>(k)+1;
			m_tree.depth.at<uchar>(K+1,0) = m_tree.depth.at<int>(k)+1;
			K=K+2;														/* adding two more nodes, left&right */
		}
		else
		{
			if(m_debug)
				cout<<"--------> no spliting <-----------"<<endl;
		}

		k++;
	}

	/* ############################# training result ############################# */
	/*  crop the infos , only need top K elements */
	m_tree.child    = m_tree.child.rowRange(0,K);
	m_tree.depth    = m_tree.depth.rowRange(0,K);
	m_tree.fids     = m_tree.fids.rowRange(0,K);
	m_tree.hs       = m_tree.hs.rowRange(0,K);
	m_tree.thrs     = m_tree.thrs.rowRange(0,K);
	m_tree.weights  = m_tree.weights.rowRange(0,K);
	errs			= errs.rowRange(0,K);

	/*  convert hs to label info, from loglikelihood to lable {1,-1} */
	/*  remains double for adaboosting training */
	convertHsToDouble();

	/*  computing the weighted error */
	m_error = 0;
	for(int i=0;i<m_tree.child.rows;i++)
	{
		if( m_tree.child.at<int>(i,0) == 0 )
			m_error += m_tree.weights.at<double>(i,0)*errs.at<double>(i,0);
	}
	
	if( m_error > 0.5)
	{
		cout<<"fatal error, train error should not be greater than 0.5, causing Adaboost training crash "<<endl;
		return false;
	}
	return true;
}

double binaryTree::getTrainError() const
{
	return m_error;
}



template< class T> 
bool  _apply( double* inds,				/* out: predicted label 1 or -1 */
			 const T *data,				/* in : data, column vector, but opencv stores data row by row in memory*/
			 const double *thrs,		/* in : thresholds  */
			 const int *fids,			/* in : feature index vector */
			 const int *child,			/* in : child index  */
			 const double *hs,			/* in : label info */
			 int number_of_samples )	/* in : feature dimension, only used for error check*/
{
	int Nthreads = std::min( 16, omp_get_max_threads());
	//#pragma omp parallel for num_threads(Nthreads)
	for ( int i=0;i<number_of_samples;i++ )
	{
		int k = 0;
		while( child[k] )					/*  not leaf node */
		{
			if( data[ fids[k]*number_of_samples +i] < thrs[k] )
			{
				k = child[k];				/* left node */
			}
			else
			{
				k = child[k]+1;				/* right node */
			}
		}
		/*  double format */
		inds[i] = hs[k];
	}
}


bool binaryTree::Apply( const Mat &inputData, Mat &predictedLabel )	 const	/* input  featuredim x number_of_sample, column vector*/
{
	if(!inputData.isContinuous() ||  inputData.channels()!=1 )
	{
		cout<<"in function Apply : please make the input data continuous and only single channel is supported"<<endl;
		return false;
	}
	if(m_tree.child.empty() || m_tree.depth.empty() || m_tree.fids.empty() || m_tree.hs.empty() || m_tree.thrs.empty() || m_tree.weights.empty())
	{
		cout<<"in function Apply : tree not ready, in function Apply "<<endl;
		return false;
	}

	if( !m_tree.child.isContinuous() || !m_tree.depth.isContinuous() || !m_tree.fids.isContinuous() || 
			!m_tree.hs.isContinuous() || !m_tree.thrs.isContinuous() || !m_tree.weights.isContinuous())
	{
		cout<<" tree model is not continuous, will result errors later "<<endl;
		return false;
	}

	/* prepare the output matrix */
	int number_of_samples = inputData.cols;
	predictedLabel = Mat::zeros( number_of_samples, 1, CV_64F);


	/* apply the decision tree to the data */
	if( inputData.type() == CV_8U)
	{
		_apply( (double*)predictedLabel.data, (uchar*)inputData.data, (double*)m_tree.thrs.data,
				(int*)m_tree.fids.data, (int*)m_tree.child.data, (double*)m_tree.hs.data, number_of_samples);
	}
	else if( inputData.type() == CV_32F)
	{
		_apply( (double*)predictedLabel.data, (float*)inputData.data, (double*)m_tree.thrs.data,
				(int*)m_tree.fids.data, (int*)m_tree.child.data, (double*)m_tree.hs.data, number_of_samples);

	}
	else if( inputData.type() == CV_32S)
	{
		_apply( (double*)predictedLabel.data, (int*)inputData.data, (double*)m_tree.thrs.data,
				(int*)m_tree.fids.data, (int*)m_tree.child.data, (double*)m_tree.hs.data, number_of_samples);

	}
	else if( inputData.type() == CV_64F)
	{
		_apply( (double*)predictedLabel.data, (double*)inputData.data, (double*)m_tree.thrs.data,
				(int*)m_tree.fids.data, (int*)m_tree.child.data, (double*)m_tree.hs.data, number_of_samples);

	}
	else
	{
		cout<<"in function Apply :unsupported data type, Should be one channel, CV_8U, CV_32F, CV_32S, CV_64F" <<endl;
		return false;
	}
	return true;
}

void binaryTree::scaleHs( double factor )
{
	m_tree.hs = m_tree.hs*factor;
}



const biTree* binaryTree::getTree() const
{
	return &m_tree;
}


void binaryTree::showTreeInfo() const
{
	cout<<"-------------------------------------------------tree information ---------------------------------------------------"<<endl;
	cout<<"depth                "<<m_tree.depth<<endl;
	cout<<"threshold info       "<<m_tree.thrs<<endl;
	cout<<"selected feature     "<<m_tree.fids<<endl;
	cout<<"child info           "<<m_tree.child<<endl;
	cout<<"hs info              "<<m_tree.hs<<endl;
	cout<<"weight info          "<<m_tree.weights<<endl;
	cout<<"weighted error is    "<<m_error<<endl;
	cout<<"----------------------------------------------------------------------------------------------------------------------"<<endl;
}

bool binaryTree::setTreeModel( const biTree& model )		/*  in : model */
{
	int number_of_element = model.fids.rows;
	if( number_of_element != model.thrs.rows ||
			number_of_element != model.child.rows ||
			number_of_element != model.hs.rows ||
			number_of_element != model.weights.rows ||
			number_of_element != model.depth.rows)
	{
		cout<<"Model file incorrect "<<endl;
		return false;
	}
	m_tree.fids = model.fids;
	m_tree.depth = model.depth;
	m_tree.hs = model.hs;
	m_tree.weights = model.weights;
	m_tree.thrs = model.thrs;
	m_tree.child = model.child;
	return true;
}
