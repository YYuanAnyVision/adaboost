#include <iostream>
#include <vector>
#include <algorithm>
#include "opencv2/highgui/highgui.hpp"
#include "Adaboost.hpp"

using namespace std;
using namespace cv;

bool Adaboost::Train(	const Mat &neg_data,				/* in : neg data format-> featuredim x number0*/
						const Mat &pos_data,				/* in : pos data format-> featuredim x number1*/
						const int &nWeaks,					/* in : how many weak classifiers( decision tree) are used  */
						const tree_para &treepara) 		/* in : parameter for the decision tree */
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

	m_trees.resize( nWeaks);

	/*  train each weak classifier */
	for ( int c=0;c<nWeaks ; c++) 
	{
		binaryTree bt; bt.SetDebug(m_debug);
		if(! bt.Train( neg_data, pos_data, treepara ) )
		{
			cout<<"in fuction Adaboost:Train, error training tree No "<<c<<endl;
			return false;
		}
		
		/*  apply the training data on the model */
		Mat predicted_label_neg, predicted_label_pos;
		bt.Apply( neg_data, predicted_label_neg);
		bt.Apply( pos_data, predicted_label_pos);

		doubel alpha = 1; double error = bt.getTrainError();
		alpha = std::max( -5, std::min( 5, 0.5*std::log( 1-error)/error));

		/*  ----------- stopping early ----------- */
		if( alpha <= 0)
		{
			cout<<"stopping early ..."<<endl;
			break;
		}
		
	}
	return false;
}

