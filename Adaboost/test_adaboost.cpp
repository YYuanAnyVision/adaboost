#include <iostream>
#include "opencv2/highgui/highgui.hpp"

#include "../binaryTree/binarytree.hpp"
#include "Adaboost.hpp"

using namespace std;
using namespace cv;

int main( int argc, char** argv)
{
	FileStorage fs;
	fs.open( "../../data/train_neg.xml" , FileStorage::READ);
	Mat train_neg, train_pos, test_pos, test_neg;
	fs["matrix"]>>train_neg;
	fs.release();

	fs.open( "../../data/train_pos.xml", FileStorage::READ);
	fs["matrix"] >>train_pos;
	fs.release();

	fs.open( "../../data/test_pos.xml", FileStorage::READ);
	fs["matrix"] >>test_pos;
	fs.release();

	fs.open( "../../data/test_neg.xml", FileStorage::READ);
	fs["matrix"] >>test_neg;
	fs.release();

	train_pos = train_pos.t();
	train_neg = train_neg.t();
	test_neg = test_neg.t();
	test_pos = test_pos.t();
	
	cout<<"train data info: "<<endl;
	cout<<"pos data  dimension : "<<train_pos.rows<<" number : "<<train_pos.cols<<endl;
	cout<<"neg data  dimension : "<<train_neg.rows<<" number : "<<train_neg.cols<<endl;

	cout<<"test data info: "<<endl;
	cout<<"pos data dimension : "<<test_pos.rows<<" number : "<<test_pos.cols<<endl;
	cout<<"neg data dimension : "<<test_neg.rows<<" number : "<<test_neg.cols<<endl;

	Adaboost ab; ab.SetDebug( false );
	int number_n_weak = 256;

	double t = getTickCount();

	tree_para train_paras;
	train_paras.nBins = 128;
	train_paras.maxDepth = 2;

	ab.Train( train_neg, train_pos, number_n_weak, train_paras);
	t = (double)getTickCount() - t;
	cout<<"time consuming is "<<t/(double)getTickFrequency()<<"s, training "<<number_n_weak<<" weak classifiers "<<endl;
	ab.saveModel("ttab.xml");

	cout<<"Load and test the model"<<endl;
	ab.loadModel( "ttab.xml");

	cout<<"test Apply function :"<<endl;
	Mat predicted_label0, predicted_label1;
	ab.ApplyLabel( test_neg, predicted_label0);
	ab.ApplyLabel( test_pos, predicted_label1);

	
	double fp = 0;
	double fn = 0;
	for(int c=0;c<predicted_label0.rows;c++)
		fp += (predicted_label0.at<int>(c,0) > 0?1:0);
	fp /= predicted_label0.rows;

	for(int c=0;c<predicted_label1.rows;c++)
		fn += (predicted_label1.at<int>(c,0) < 0?1:0);
	fn /= predicted_label1.rows;


	/*  fp and fn should be around 0.12-0.13  */
	cout<<"Results on the training data is :"<<endl;
	cout<<"--> False Positive is "<<fp<<endl;
	cout<<"--> False Negative is "<<fn<<endl;


	return 0;
}
