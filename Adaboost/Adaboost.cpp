#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include "opencv2/highgui/highgui.hpp"
#include "Adaboost.hpp"

using namespace std;
using namespace cv;

bool Adaboost::Train(	const Mat &neg_data,				/* in : neg data format-> featuredim x number0*/
						const Mat &pos_data,				/* in : pos data format-> featuredim x number1*/
						const int &nWeaks,					/* in : how many weak classifiers( decision tree) are used  */
						const tree_para &treepara)			/* in : parameter for the decision tree */
{
	if( neg_data.rows != pos_data.rows || neg_data.type() != pos_data.type())
	{
		cout<<"In function Adaboost:Train : neg_data and pos_data should be the same type, and having the same rows( column feature vector)"<<endl;
		return false;
	}
	if(  neg_data.empty() || pos_data.empty() )
	{
		cout<<"In function Adaboost:Train : data empty "<<endl;
		return false;
	}

	/*  using the specific format, carrying weight and other information  */
	data_pack train_pack;
	train_pack.neg_data = neg_data;
	train_pack.pos_data = pos_data;

	/*  infos about the data  */
	int number_neg_samples = neg_data.cols;
	int number_pos_samples = pos_data.cols;
	int feature_dim		   = neg_data.rows;

	/*  scores about the sample */
	Mat H0 = Mat::zeros( number_neg_samples, 1, CV_64F);
	Mat H1 = Mat::zeros( number_pos_samples, 1, CV_64F);
	
	/*  infos about training , errs and losses */
	Mat losses = Mat::zeros( nWeaks, 1, CV_64F);
	Mat errs   = Mat::zeros( nWeaks, 1, CV_64F);

	m_trees.reserve( nWeaks);
	/*  train each weak classifier */
	for ( int c=0;c<nWeaks ; c++) 
	{
		if(m_debug)
			cout<<"\n\n\n############################################## round "<<c<<" ##########################################"<<endl;
		binaryTree bt; bt.SetDebug(true);
		if(! bt.Train( train_pack , treepara ) )
		{
			cout<<"in fuction Adaboost:Train, error training tree No "<<c<<endl;
			return false;
		}

		/*  apply the training data on the model */
		Mat h0, h1;										/*  predicted labels */
		bt.Apply( neg_data, h0);
		bt.Apply( pos_data, h1);

		double alpha = 1; 
		double error = bt.getTrainError();
		alpha = std::max( -5.0, std::min(  5.0, 0.5*std::log((1-error)/error)  ));
		if(m_debug)
			cout<<"alpha is "<<alpha<<" , error is "<<error<<endl;

		/*  ----------- stopping early ----------- */
		if( alpha <= 0)
		{
			cout<<"stopping early ..."<<endl;
			break;
		}
		bt.scaleHs( alpha );

		if(m_debug)
		{
			cout<<"Tree No "<<c<<" Infos: "<<endl;
			bt.showTreeInfo();
		}

		/* update cumulative scores H and weights */
		H0 = H0 + alpha*h0;
		H1 = H1 + alpha*h1;

		cv::exp( H0, train_pack.wts0 );       train_pack.wts0 = train_pack.wts0/(2*number_neg_samples);
		cv::exp( -1.0*H1, train_pack.wts1 );  train_pack.wts1 = train_pack.wts1/(2*number_pos_samples);
		
		double loss = (cv::sum( train_pack.wts0))[0] + (cv::sum(train_pack.wts1))[0];
		m_trees.push_back( bt );
		errs.at<double>(c,0) = bt.getTrainError();
		losses.at<double>(c,0) = loss;
	}

	/* --------------------------  output debug information ------------------------------ */
	if(m_debug)
	{
		for ( int c=0;c<errs.rows ;c++ ) 
		{

			cout <<setprecision(8)<< "errs in weak learner "<<c<<"\t err="<<errs.at<double>(c,0)<<"\t loss ="<<losses.at<double>(c,0)<<endl;
		}
	}

	return false;
}


void Adaboost::SetDebug( bool d )
{
	m_debug = d;
}
