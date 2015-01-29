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

	//Adaboost ab;
	//ab.loadModel("ab_3.xml");

	//softcascade sc;
	//sc.Load("scmodel.xml");

	//FileStorage fs("neg_test_data.xml", FileStorage::READ);
	//Mat neg_test_data;
	//fs["neg_data"]>>neg_test_data;

	//Mat for_test_data = neg_test_data.col(0);
	//Mat sc_only; for_test_data.copyTo( sc_only);
	//double sc_score = 0;
	//sc.Predict( (const float*)sc_only.data, sc_score);

	//Mat ab_test; for_test_data.copyTo(ab_test ); Mat ad_predict;
	//ab.Apply( ab_test, ad_predict);
	//cout<<"sc score is "<<sc_score<<endl;
	//cout<<"ab score is "<<ad_predict.at<double>(0,0)<<endl;

	//for( int c=0;c<for_test_data.rows;c++)
	//{
	//	cout<<"fea index "<<c<<" "<<for_test_data.at<float>(c,0)<<endl;
	//}
    softcascade sc;
    sc.Load("for_test_sc.xml");
    //Mat test_img = imread(argv[1]);
    //vector<Rect> det_rects;
    //vector<double> det_confs;
    //sc.detectMultiScale( test_img, det_rects, det_confs );
    //for( int c =0; c<det_rects.size(); c++)
    //{
    //    rectangle( test_img, det_rects[c], Scalar(255,0,0) );
    //}
    //imshow("test", test_img);
    //waitKey(0);
    sc.visulizeFeature();
	return 0;
}
