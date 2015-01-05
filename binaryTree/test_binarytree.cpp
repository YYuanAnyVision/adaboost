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

	data_pack train_pack;
	train_pack.neg_data = train_neg;
	train_pack.pos_data = train_pos;

	double t = getTickCount();
	bt.Train( train_pack, tree_para());
	cout<<"training time : "<<((double)getTickCount() - t)/double(getTickFrequency())<<endl;

	bt.showTreeInfo();
	//bt.scaleHs( 0.5);
	//bt.showTreeInfo();
	//cout<<"weight 0 after Train function is "<<train_pack.wts0<<endl;
	//cout<<"weight 1 after Train function is "<<train_pack.wts1<<endl;

	//cout<<"quantization info is "<<endl;
	//cout<<"Xmin  \n"<<train_pack.Xmin<<endl;
	//cout<<"Xmax  \n"<<train_pack.Xmax<<endl;
	//cout<<"Xstep \n"<<train_pack.Xstep<<endl;

	Mat p;
	bt.Apply( train_neg, p);
	//cout<<"predicted label for train_neg data is "<<p<<endl<<endl;

	bt.Apply( train_pos, p);
	//cout<<"predicted label for train_pos data is "<<p<<endl;


	/*  train again using weights and quantization from previous step */
	t = getTickCount();
	bt.Train( train_pack, tree_para());
	cout<<"training time : "<<((double)getTickCount() - t)/double(getTickFrequency())<<endl;
	
	return 0;
}
