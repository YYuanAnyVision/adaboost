#include <iostream>
#include <vector>
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "softcascade.hpp"
#include "../Adaboost/Adaboost.hpp"
#include "../misc/misc.hpp"

using namespace std;
using namespace cv;


int main( int argc, char** argv)
{
	vector<Adaboost> abs;
	Adaboost ab1,ab2,ab3;
	ab1.loadModel( "t_dep2.xml");
	//ab2.loadModel( "t_dep2.xml");
	//ab3.loadModel( "t_dep2.xml");
	abs.push_back( ab1 );
	//abs.push_back( ab2 );
	//abs.push_back( ab3);
	
	softcascade sc;
	sc.Combine( abs);


	Mat test_pos,test_neg;
	FileStorage fs;
	fs.open( "../../data/test_pos.xml", FileStorage::READ);
	fs["matrix"] >>test_pos;
	fs.release();

	fs.open( "../../data/test_neg.xml", FileStorage::READ);
	fs["matrix"] >>test_neg;
	fs.release();

	cout<<"test_neg, number of sample "<<test_pos.rows<<" feature dim "<<test_pos.cols<<endl;

	int fn = 0;
	int fp = 0;
	for ( int c=0; c<test_pos.rows;c++ ) 
	{
		double h = 0;
		double* pp = test_pos.ptr<double>(c);
		sc.Predict( pp, h );
		if( h < 0)
			fn++;
	}
	cout<<"False Negative is "<<fn*1.0/test_pos.rows<<endl;

	for ( int c=0; c<test_neg.rows;c++ ) 
	{
		double h = 0;
		double* pp = test_neg.ptr<double>(c);
		sc.Predict( pp, h );
		if( h > 0)
			fp++;
	}
	cout<<"False Positive is "<<fp*1.0/test_neg.rows<<endl;
	return 0;
}
