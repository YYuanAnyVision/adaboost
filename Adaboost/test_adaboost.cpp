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
	Mat train_neg, train_pos;
	fs["matrix"]>>train_neg;
	fs.release();

	fs.open( "../../data/train_pos.xml", FileStorage::READ);
	fs["matrix"] >>train_pos;
	fs.release();

	train_pos = train_pos.t();
	train_neg = train_neg.t();
		
	cout<<"pos data size "<<train_pos.rows<<" "<<train_pos.cols<<endl;
	cout<<"neg data size "<<train_neg.rows<<" "<<train_neg.cols<<endl;

	Adaboost ab; ab.SetDebug( true );

	double t = getTickCount();
	ab.Train( train_neg, train_pos, 20, tree_para());
	t = (double)getTickCount() - t;
	cout<<"time consuming is "<<t/(double)getTickFrequency()<<endl;

	return 0;
}
