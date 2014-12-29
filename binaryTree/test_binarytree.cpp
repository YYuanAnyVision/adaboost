#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "binarytree.hpp"

using namespace std;
using namespace cv;

int main( int argc, char** argv)
{
	binaryTree bt;
	bt.SetDebug( true );
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
	double t = getTickCount();
	bt.Train( train_neg, train_pos, tree_para());
	cout<<"training time : "<<((double)getTickCount() - t)/double(getTickFrequency())<<endl;
	Mat p;
	bt.Apply( train_neg, p);
	cout<<"predicted label for train_neg data is "<<p<<endl;
	return 0;
}
