#include <iostream>
#include <vector>
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "softcascade.hpp"
#include "../Adaboost/Adaboost.hpp"
#include "../misc/misc.hpp"
#include "detect_check.h"

using namespace std;
using namespace cv;


int main( int argc, char** argv)
{
    softcascade sc;
    if(!sc.Load("for_test_sc.xml"))
    {
        cout<<"Can not load the model "<<endl;
        return -1;
    }
    
    string test_img_folder = "/home/yuanyang/Workspace/INRIA/Test/pos/";
    string test_img_gt  = "/home/yuanyang/Workspace/INRIA/Test/AnnotTest/";

    double _hit = 0;
    double _FPPI =0;
    detect_check<softcascade> dc;
    dc.set_path( test_img_folder, test_img_gt, "", true);
    dc.set_parameter( Size(41,100), Size(1000,2000), 1.2, 1, 0);

    //vector<double> hits;
    //vector<double> fppis;
    //dc.generate_roc( fhog_sc, fppis, hits);

    dc.test_detector( sc, _hit, _FPPI);
    dc.get_stat_on_missed();
    cout<<"Results : \nHit : "<<_hit<<endl<<"FPPI : "<<_FPPI<<endl;
	return 0;
}
