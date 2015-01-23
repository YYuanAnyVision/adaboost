#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/contrib/contrib.hpp"

#include "../binaryTree/binarytree.hpp"
#include "Adaboost.hpp"

using namespace std;
using namespace cv;

int main( int argc, char** argv)
{

    /* -------------check the training process using matlab data , result--> good ----------- */
    cout<<"Loading data ..."<<endl;
    FileStorage fs;

    fs.open("X0train_first.xml", FileStorage::READ);
    Mat train_neg;
    fs["matrix"]>>train_neg;
    fs.release();

    fs.open("X1train.xml", FileStorage::READ);
    Mat train_pos;
    fs["matrix"]>>train_pos;
    fs.release();

    fs.open("X1test.xml", FileStorage::READ);
    Mat test_pos;
    fs["matrix"]>>test_pos;
    fs.release();

    fs.open("X0test.xml", FileStorage::READ);
    Mat test_neg;
    fs["matrix"]>>test_neg;
    fs.release();



    cout<<"Loading data done"<<endl;
    cout<<"pos data size "<<train_pos.size()<<endl;
    cout<<"neg data size "<<train_neg.size()<<endl;

	Adaboost ab; ab.SetDebug( false );
	int number_n_weak = 2048;

	double t = getTickCount();

	tree_para train_paras;
	train_paras.nBins = 256;
	train_paras.maxDepth = 2;
    train_paras.fracFtrs = 0.0625;

	/*  Train function will change the data  */
	ab.Train( train_neg, train_pos, number_n_weak, train_paras);
	t = (double)getTickCount() - t;
	cout<<"Adaboost decision tree : time consuming is "<<t/(double)getTickFrequency()<<"s, training "<<number_n_weak<<" weak classifiers "<<endl;

    cout<<"Now testing "<<endl;
	Mat predicted_label0, predicted_label1;
	ab.Apply( test_neg, predicted_label0);
	ab.Apply( test_pos, predicted_label1);

    double avg_neg_dis = 0;
    double avg_pos_dis = 0;
	double fp = 0;
	double fn = 0;
	for(int c=0;c<predicted_label0.rows;c++)
    {
        avg_neg_dis += predicted_label0.at<double>(c,0);
		fp += (predicted_label0.at<double>(c,0) > 0?1:0);
    }
	fp /= predicted_label0.rows;
    avg_neg_dis /= predicted_label0.rows;

	for(int c=0;c<predicted_label1.rows;c++)
    {
        avg_pos_dis += predicted_label1.at<double>(c,0);
		fn += (predicted_label1.at<double>(c,0) < 0?1:0);
    }
    avg_pos_dis /= predicted_label1.rows;
	fn /= predicted_label1.rows;


	/*  fp and fn should be around 0.12-0.13  */
	cout<<"Results on the training data is :"<<endl;
	cout<<"--> False Positive is "<<fp<<endl;
	cout<<"--> False Negative is "<<fn<<endl;
    cout<<"avg_neg_dis is "<<avg_neg_dis<<endl;
    cout<<"avg_pos_dis is "<<avg_pos_dis<<endl;


    ///* --------------------------- compare the results with svm -------------------- */
	//FileStorage fs;
    //fs.open( "../../data/train_neg.xml" , FileStorage::READ);
	//Mat train_neg, train_pos, test_pos, test_neg;
	//fs["matrix"]>>train_neg;
	//fs.release();

    //fs.open( "../../data/train_pos.xml", FileStorage::READ);
	//fs["matrix"] >>train_pos;
	//fs.release();

    //fs.open( "../../data/test_pos.xml", FileStorage::READ);
	//fs["matrix"] >>test_pos;
	//fs.release();

    //fs.open( "../../data/test_neg.xml", FileStorage::READ);
	//fs["matrix"] >>test_neg;
	//fs.release();

	//train_pos = train_pos.t();
	//train_neg = train_neg.t();
	//test_neg = test_neg.t();
	//test_pos = test_pos.t();
	//
	//cout<<"train data info: "<<endl;
	//cout<<"pos data  dimension : "<<train_pos.rows<<" number : "<<train_pos.cols<<endl;
	//cout<<"neg data  dimension : "<<train_neg.rows<<" number : "<<train_neg.cols<<endl;

	//cout<<"test data info: "<<endl;
	//cout<<"pos data dimension : "<<test_pos.rows<<" number : "<<test_pos.cols<<endl;
	//cout<<"neg data dimension : "<<test_neg.rows<<" number : "<<test_neg.cols<<endl;

	//Adaboost ab; ab.SetDebug( false );
	//int number_n_weak = 256;

	//double t = getTickCount();

	//tree_para train_paras;
	//train_paras.nBins = 256;
	//train_paras.maxDepth = 2;
    //train_paras.fracFtrs = 1;

	///*  Train function will change the data  */
	//ab.Train( train_neg, train_pos, number_n_weak, train_paras);
	//t = (double)getTickCount() - t;
	//cout<<"Adaboost decision tree : time consuming is "<<t/(double)getTickFrequency()<<"s, training "<<number_n_weak<<" weak classifiers "<<endl;
	//ab.saveModel("ttab.xml");

	//cout<<"Load and test the model"<<endl;
	//ab.loadModel( "ttab.xml");

	//cout<<"test Apply function :"<<endl;
	//Mat predicted_label0, predicted_label1;
	//ab.ApplyLabel( test_neg, predicted_label0);
	//ab.ApplyLabel( test_pos, predicted_label1);

	////cout<<"predicted label for negative sample is "<<predicted_label0<<endl;
	//
	//double fp = 0;
	//double fn = 0;
	//for(int c=0;c<predicted_label0.rows;c++)
	//	fp += (predicted_label0.at<int>(c,0) > 0?1:0);
	//fp /= predicted_label0.rows;

	//for(int c=0;c<predicted_label1.rows;c++)
	//	fn += (predicted_label1.at<int>(c,0) < 0?1:0);
	//fn /= predicted_label1.rows;


	///*  fp and fn should be around 0.12-0.13  */
	//cout<<"Results on the training data is :"<<endl;
	//cout<<"--> False Positive is "<<fp<<endl;
	//cout<<"--> False Negative is "<<fn<<endl;

	//cout<<"the nodes of the tree is "<<ab.getTreesNodes()<<endl;
	//cout<<"tht max number of nodes is "<<ab.getMaxNumNodes()<<endl;


	///*  test the getTrees function */
	//const vector<binaryTree> re_trees = ab.getTrees();
	//const biTree *ptr = re_trees[0].getTree();
	//cout<<"sample of fids is "<<(*ptr).fids<<endl;

	///*  compared with opencv' svm  */
	//cout<<"=================== compare with linear svm(RBF) ====================  "<<endl;
	//train_pos = train_pos.t();
	//train_neg = train_neg.t();
	//test_neg = test_neg.t();
	//test_pos = test_pos.t();
	//
	//Mat traindata = Mat::zeros( train_pos.rows+train_neg.rows, train_pos.cols, train_pos.type() );
	//Mat trainlabel= Mat::ones( traindata.rows, 1 , CV_32S);

	//train_pos.copyTo( traindata.rowRange(0,train_pos.rows) );
	//train_neg.copyTo( traindata.rowRange( train_pos.rows, traindata.rows));
	//trainlabel.rowRange(train_pos.rows, traindata.rows) = -1;
	//traindata.convertTo( traindata, CV_32FC1);

	//CvSVMParams params;
	//params.svm_type = SVM::C_SVC;
	//params.C = 0.1;
    //params.kernel_type = SVM::LINEAR;
	//params.gamma = 0.1;
	//params.term_crit =  TermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-6);
	//
	//CvSVM svm;
	//cv::TickMeter tk;
	//tk.start();
	//svm.train( traindata, trainlabel, Mat(), Mat(), params );
	//tk.stop();
    //cout<<"finished svm training "<<endl;
	//cout<<"Svm training time consuming is "<<tk.getTimeSec()<<" s "<<endl;
	//
	//test_neg.convertTo( test_neg, CV_32FC1);
	//fp = 0;
	//for( int c=0;c<test_neg.rows;c++)
	//{
	//	float response = svm.predict( test_neg.row(c) );
	//	if( response > 0 )
	//		fp+=1;
	//}
	//cout<<"--> False Positive is "<<fp/test_neg.rows<<endl;

	//test_pos.convertTo( test_pos, CV_32FC1);
	//fn = 0;
	//for( int c=0;c<test_pos.rows;c++)
	//{
	//	float response = svm.predict( test_pos.row(c) );
	//	if( response < 0 )
	//		fn+=1;
	//}
	//cout<<"--> False Negative is "<<fn/test_pos.rows<<endl;

	return 0;
}
