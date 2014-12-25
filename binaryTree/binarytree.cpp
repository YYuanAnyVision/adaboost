#include <iostream>
#include "opencv2/highgui/highghui.cpp"
#include "binarytree.hpp"

using namespace cv;

bool binaryTree::checkTreeParas( const tree_para & p )		/* input parameter */
{
	if( p.nBins < 2 || p.maxDepth< 1 || p.minWeight < 0 || p.minWeight > 1 || p.nThreads < 0 || 
			p.fracFtrs >1 || p.fracFtrs<0)
		return false;
	else
		return true;
}

bool binaryTree::Train( const Mat &neg_data,			/* input, format-> featuredim x number0 */
						const Mat &pos_data,			/* input, format-> featuredim x number1 */ 
						const tree_para &paras			/* input tree paras */)
{
	/* sanity check*/
	if( neg_data.empty() || pos_data.empty() || !checkTreeParas(paras))
	{
		cout<<"input wrong format"<<endl;
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



}
