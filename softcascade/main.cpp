#include <iostream>
#include <vector>
#include <algorithm>

#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"

#include "boost/filesystem.hpp"
#include "boost/lambda/bind.hpp"

#include "../Adaboost/Adaboost.hpp"
#include "../misc/misc.hpp"
#include "softcascade.hpp"
#include "../chnfeature/Pyramid.h"

#include <omp.h>

using namespace std;
using namespace cv;

namespace bf = boost::filesystem;
namespace bl = boost::lambda;


/* extract the center part of the feature( crossponding to the target) , save it as 
 * clo in output_data(not continuous) */
void makeTrainData( vector<Mat> &in_data, Mat &output_data, Size modelDs, int shrink)
{
    assert( output_data.type() == CV_32F);
    assert( in_data[0].type() == CV_32F && in_data[0].isContinuous());

	int w_in_data = in_data[0].cols;
	int h_in_data = in_data[0].rows;

	int w_f = modelDs.width/shrink;
	int h_f = modelDs.height/shrink;
    
    assert( w_in_data > w_f && h_in_data > h_f );

    float *p_end = (float*)in_data[0].ptr() + h_in_data*w_in_data*in_data.size();

	int Nthreads = omp_get_max_threads();
	for( int c=0;c < in_data.size(); c++)
	{
        float *ptr=(float*)in_data[c].ptr() + (h_in_data - h_f)/2*w_in_data + (w_in_data - w_f)/2;
        for( int j=0;j<h_f;j++)
        {
            float *pp = ptr + j*w_in_data;
            for( int i=0;i<w_f;i++)
            {
                output_data.at<float>( c*h_f*w_f + j*w_f+i ,0) = pp[i];    
            }
        }

	}
	
}

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

                /*  adding flipped version of the image */
                Mat flipped_target;
                cv::flip( target_obj, flipped_target, 1 );
                origsamples.push_back( flipped_target );

            }
        }
    }
    else
    {
		bf::path neg_img_path(opts.negImgDir);
		int number_target_per_image = opts.nPerNeg;

		if(!bf::exists(neg_img_path))
		{
			cout<<"negative image folder path "<<neg_img_path<<" dose not exist "<<endl;
			return false;
		}
		int number_of_neg_images = 	getNumberOfFilesInDir( opts.negImgDir );
		
		/* shuffle the path */
		vector<string> neg_paths;
        bf::directory_iterator end_it; int file_counter = 0;int number_target = 0;
        for( bf::directory_iterator file_iter(neg_img_path); file_iter!=end_it; file_iter++)
		{
            string pathname = file_iter->path().string();
			neg_paths.push_back( pathname );
		}

		std::random_shuffle( neg_paths.begin(), neg_paths.end() );
		
        
		for( int c=0;c<neg_paths.size() ;c++)
		{
			vector<Rect> target_rects;

			Mat img = imread( neg_paths[c] );
			if(img.empty())
			{
				cout<<"can not read image "<<neg_paths[c]<<endl;
				return false;
			}
			/*  inf stage == 0, first time just sample the image, otherwise add the "hard sample" */
			if( stage==0 )
			{
                /*  sampling and shuffle  */
				sampleRects( number_target_per_image*2, img.size(), opts.modelDs, target_rects );
                std::random_shuffle( target_rects.begin(), target_rects.end() );

			}
			else
			{

			}
			
            /*  resize the rect to fixed widht / height ratio, for pedestrain det , is 41/100 for INRIA database */
            for ( int i=0;i<target_rects.size() && i < number_target_per_image;i++) 
            {
                target_rects[i] = resizeToFixedRatio( target_rects[i], opts.modelDs.width*1.0/opts.modelDs.height, 1); /* respect to height */
                /* grow it a little bit */
                int modelDsBig_width = std::max( 8*opts.shrink, opts.modelDsPad.width)+std::max(2, 64/opts.shrink)*opts.shrink;
                int modelDsBig_height = std::max( 8*opts.shrink,opts.modelDsPad.height)+std::max(2,64/opts.shrink)*opts.shrink;
                

                double w_ratio = modelDsBig_width*1.0/opts.modelDs.width;
                double h_ratio = modelDsBig_height*1.0/opts.modelDs.height;
                target_rects[i] = resizeBbox( target_rects[i], h_ratio, w_ratio);
                

                /* finally crop the image */
                Mat target_obj = cropImage( img, target_rects[i]);
                cv::resize( target_obj, target_obj, cv::Size(modelDsBig_width, modelDsBig_height), 0, 0, INTER_AREA);
                origsamples.push_back( target_obj );
            }
		}

        /* random shuffle and sampling  */
        if(origsamples.size() > number_to_sample)
            origsamples.resize( number_to_sample);
    }
}



    
/* detector parameter define */
int main( int argc, char** argv)
{

    /* globale paras*/
	int Nthreads = omp_get_max_threads();
    TickMeter tk;

	/*-----------------------------------------------------------------------------
	 *  Step 1 : set and check parameters
	 *-----------------------------------------------------------------------------*/
    softcascade sc;
	cascadeParameter cas_para;
    //cas_para.posGtDir  = "/mnt/disk1/data/INRIAPerson/Train/posGT/";
    cas_para.posGtDir  = "/media/yuanyang/disk1/libs/piotr_toolbox/data/Inria/train/posGt_opencv/";
    //cas_para.posImgDir = "/mnt/disk1/data/INRIAPerson/Train/pos"; 
    cas_para.posImgDir = "/media/yuanyang/disk1/libs/piotr_toolbox/data/Inria/train/pos/"; 
	//cas_para.negImgDir = "/mnt/disk1/data/INRIAPerson/Train/neg/";
	cas_para.negImgDir = "/media/yuanyang/disk1/libs/piotr_toolbox/data/Inria/train/neg/";
    sc.setParas( cas_para);
    

    vector<Mat> neg_samples;
    vector<Mat> neg_origsamples;
    cout<<"reading and sampling pos&neg image patches "<<endl;
    tk.start();
    sampleWins( sc, 0, false, neg_samples, neg_origsamples);
    
    /*  show samples  */
    //for ( int c=0;c<neg_origsamples.size();c++) 
    //{
    //    imshow("sample", neg_origsamples[c] );
    //    imwrite("sample.png", neg_origsamples[c]);
    //    waitKey(0);
    //}
	
    vector<Mat> pos_samples;
    vector<Mat> pos_origsamples;
    sampleWins( sc, 0, true, pos_samples, pos_origsamples);
    tk.stop();
	cout<<"# sampling time consuming is "<<tk.getTimeSec()<<" seconds "<<endl;
	cout<<"neg sample number -> "<<neg_origsamples.size()<<endl;
	cout<<"pos sample number -> "<<pos_origsamples.size()<<endl;


    /*  ------------------------- test Adaboost for one stage --------------------------- */
    Size modelDsPad = cas_para.modelDsPad;
    int n_channels = 10;
    int n_shrink = 4;
    int final_feature_dim = modelDsPad.width/n_shrink*modelDsPad.height/n_shrink*n_channels;

    Mat pos_train_data = Mat::zeros( final_feature_dim, pos_origsamples.size(), CV_32F);
    Mat neg_train_data = Mat::zeros( final_feature_dim, neg_origsamples.size(), CV_32F);

    feature_Pyramids ff1;
    
    tk.reset();tk.start();
    cout<<"computing features for file "<<endl;

    #pragma omp parallel for num_threads(Nthreads)
    for ( int c=0;c<pos_origsamples.size();c++) 
    {
        vector<Mat> feas;
        ff1.computeChannels( pos_origsamples[c], feas );
        Mat tmp = pos_train_data.col(c);
        makeTrainData( feas, tmp , cas_para.modelDsPad, 4);
        if(c==0)
        {
            cout<<"channels is "<<feas.size()<<endl;
            cout<<"feature dim col "<<feas[c].cols<<" "<<feas[c].rows<<" final size "<<endl;
        }
    }

    #pragma omp parallel for num_threads(Nthreads)
    for ( int c=0;c<neg_origsamples.size();c++)
    {
        vector<Mat> feas;
        ff1.computeChannels( neg_origsamples[c], feas );
        Mat tmp = neg_train_data.col(c);
        makeTrainData( feas, tmp , cas_para.modelDsPad, 4);
    }
    tk.stop();
    cout<<" #time consuming "<<tk.getTimeSec()<<" seconds, computing features done for "<<neg_origsamples.size() + pos_origsamples.size()<<" targets "<<endl;


    Adaboost ab;ab.SetDebug(false);
    tree_para ad_para;
    ad_para.nBins = 256;
    ad_para.maxDepth = 2;
    ad_para.fracFtrs = 0.0625;

    /* check adaboost training*/
    ab.Train( neg_train_data, pos_train_data, 64, ad_para );
    vector<Adaboost> v_ab;
    v_ab.push_back( ab);
    sc.Combine( v_ab );

    
    /*  check softcascade combining  */
    pos_train_data = pos_train_data.t();
    double h = 0;
    double fn =0;
    for ( int c=0;c<pos_train_data.rows;c++ ) 
    {
        h = 0;
        float *pp = pos_train_data.ptr<float>(c);
		sc.Predict( pp, h );
		if( h < 0)
			fn++;
    }
    cout<<"fn is "<<fn/pos_train_data.rows<<endl;


    /* test softcascade Apply function */
    Mat test_apply = imread("test_apply.png");

    vector<Mat> f_chns;
    vector<Rect> rs;vector<double> conf;

    tk.reset();tk.start();
    ff1.computeChannels( test_apply, f_chns);
    sc.Apply( f_chns[0], rs, conf);
    tk.stop();
    cout<<"# det all, time consuming "<<tk.getTimeSec()<<endl;
    cout<<"det size "<<rs.size()<<endl;
    for(int c=0;c<rs.size(); c++)
    {
        if( conf[c]> 3)
            rectangle( test_apply, rs[c], Scalar(255,0,128));
    }

    imshow( "s", test_apply);
    waitKey(0);
    

    /*  ------------------------- test Adaboost for one stage --------------------------- */


	/*-----------------------------------------------------------------------------
	 *  Step 2 : iterate bootstraping and training
	 *-----------------------------------------------------------------------------*/
	for( int stage=0;stage<cas_para.nWeaks.size();stage++)
	{
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
