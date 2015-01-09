#include <iostream>
#include <vector>
#include <algorithm>

#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "boost/filesystem.hpp"
#include "boost/lambda/bind.hpp"

#include "../Adaboost/Adaboost.hpp"
#include "../misc/misc.hpp"
#include "softcascade.hpp"

using namespace std;
using namespace cv;

namespace bf = boost::filesystem;
namespace bl = boost::lambda;

size_t getNumberOfFilesInDir( string in_path )
{
    bf::path c_path(in_path);   
    if( !bf::exists(c_path))
        return -1;
    if( !bf::is_directory(c_path))
        return -1;

    int cnt = std::count_if(
        bf::directory_iterator( c_path ),
        bf::directory_iterator(),
        bl::bind( static_cast<bool(*)(const bf::path&)>(bf::is_regular_file), 
        bl::bind( &bf::directory_entry::path, bl::_1 ) ) );
    return cnt;
}


bool sampleWins(    const softcascade &sc, 	    /*  in: detector */
                    int stage, 			        /*  in: stage */
                    bool isPositive,            /*  in: true->sample positive, false -> sample negative */
                    vector<Mat> &samples,       /* out: target objects, flipped */
                    vector<Mat> &origsamples)   /* out: original target */
{
    origsamples.clear();
    samples.clear();

    cascadeParameter opts = sc.getParas();
    int number_to_sample = 0;
    if(isPositive)
        number_to_sample = opts.nPos;
    else
        number_to_sample = opts.nNeg;
    
    if(isPositive)
    {
        bf::path pos_img_path( opts.posImgDir );
        bf::path pos_gt_path( opts.posGtDir );

        if( !bf::exists( pos_img_path) || !bf::exists(pos_gt_path))
        {
            cout<<"pos img or gt path does not exist!"<<endl;
            cout<<"check "<<pos_img_path<<"  and "<<pos_gt_path<<endl;
            return false;
        }
        int number_pos_img = getNumberOfFilesInDir( opts.posGtDir );
        /* iterate the folder*/
        bf::directory_iterator end_it; int file_counter = 0;int number_target = 0;
        for( bf::directory_iterator file_iter(pos_img_path); file_iter!=end_it; file_iter++)
        {
            /*  number_to_sample < 0  means inf, don't stop */
            if( number_to_sample > 0 && number_target>number_to_sample )
                break;

            bf::path s = *(file_iter);
            string basename = bf::basename( s );
            string pathname = file_iter->path().string();
            string extname  = bf::extension( s );
            
            cout<<"reading and cropping image "<<pathname<<endl;

            /* read the gt according to the image name */
            string gt_file_path = opts.posGtDir + basename + ".txt";
            Mat im = imread(pathname);
            if(im.empty())
            {
                cout<<"can not read image file "<<pathname<<endl;
                return false;
            }

            vector<Rect> target_rects;
            FileStorage fst( gt_file_path, FileStorage::READ | FileStorage::FORMAT_XML);
            if(!fst.isOpened())
            {
                cout<<"can not read gt file "<<gt_file_path<<endl;
                return false;
            }
            fst["boxes"]>>target_rects;
            fst.release();

            /*  resize the rect to fixed widht / height ratio, for pedestrain det , is 41/100 for INRIA database */
            for ( int i=0;i<target_rects.size();i++) 
            {
                target_rects[i] = resizeToFixedRatio( target_rects[i], opts.modelDs.width*1.0/opts.modelDs.height, 1); /* respect to height */
                /* grow it a little bit */
                int modelDsBig_width = std::max( 8*opts.shrink, opts.modelDsPad.width)+std::max(2, 64/opts.shrink)*opts.shrink;
                int modelDsBig_height = std::max( 8*opts.shrink,opts.modelDsPad.height)+std::max(2,64/opts.shrink)*opts.shrink;
                

                double w_ratio = modelDsBig_width*1.0/opts.modelDs.width;
                double h_ratio = modelDsBig_height*1.0/opts.modelDs.height;
                target_rects[i] = resizeBbox( target_rects[i], h_ratio, w_ratio);
                

                /* finally crop the image */
                Mat target_obj = cropImage( im, target_rects[i]);
                cv::resize( target_obj, target_obj, cv::Size(modelDsBig_width, modelDsBig_height), 0, 0, INTER_AREA);
                origsamples.push_back( target_obj );
            }

            
        }
    }
    else
    {

    }
}



    
/* detector parameter define */
int main( int argc, char** argv)
{

	/*-----------------------------------------------------------------------------
	 *  Step 1 : set and check parameters
	 *-----------------------------------------------------------------------------*/
    softcascade sc;
	cascadeParameter cas_para;
    cas_para.posGtDir  = "/media/yuanyang/disk1/libs/piotr_toolbox/data/Inria/train/posGt_opencv/";
    cas_para.posImgDir = "/media/yuanyang/disk1/libs/piotr_toolbox/data/Inria/train/pos/"; 
    sc.setParas( cas_para);
    

    vector<Mat> samples;
    vector<Mat> origsamples;
    sampleWins( sc, 0, true, samples, origsamples);
    cout<<"the number of samples is "<<origsamples.size()<<endl;
    
    for ( int c=0;c<origsamples.size();c++) 
    {
        imshow("sample", origsamples[c] );
        waitKey(0);
    }


	/*-----------------------------------------------------------------------------
	 *  Step 2 : iterate bootstraping and training
	 *-----------------------------------------------------------------------------*/
	for( int stage=0;stage<cas_para.nWeaks.size();stage++)
	{
		TickMeter tk;
		tk.start();
		cout<<"Training Stage No "<<stage<<endl;
		/* TODO sample positives and compute info about channels */
		//if( stages == 0)
		//{
		//}

		/* TODO compute local decorrelation filters if needed */

		/* TODO compute lambdas */

		/* TODO compute features for positives */

		/* TODO sample negatives and compute features */

		/* TODO accumulate negatives from previous stages */

		/* TODO train boosted classifiers */
		tk.stop();
		cout<<"Done Stage No "<<stage<<" , time "<<tk.getTimeSec()<<endl;
	}

}
