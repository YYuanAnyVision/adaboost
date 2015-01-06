#include <iostream>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "misc.hpp"

using namespace std;
using namespace cv;

int main( int argc, char** argv)
{
	Mat input_image  = imread("../../data/test.png");

	cout<<"input image, cols "<<input_image.cols<<" rows "<<input_image.rows<<endl;
	imshow("original", input_image);
	
	//cout<<"test resizeBbox "<<endl;
	//Rect o1(100, 100, 150, 150);
	//Rect o1_resize = resizeBbox( o1, 1, 0.5);
	//Rect o2_resize = resizeBbox( o1, 0.5, 1);
	//imshow( "before resize", input_image(o1));
	//imshow( "resize o1", input_image(o1_resize) );
	//imshow( "resize o2", input_image(o2_resize) );

	//cout<<"test using cropSize "<<endl;
	//Rect cropEnlarge( 400, 200, 200, 250);
	//imshow("cropSizeimage", cropImage(input_image, cropEnlarge));
	
	vector<Rect> pos;
	sampleRects( 30, input_image.size(), cv::Size(150,80), pos);
	cout<<"size of pos is "<<pos.size()<<endl;
	for( int c=0;c<pos.size(); c++)
	{
		cout<<"c is "<<c<<" rect is "<<pos[c]<<endl;
		imshow( "samples", input_image( pos[c] ) );
		waitKey(30);
	}

	waitKey(0);
	return 0;
}
