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
#include "../misc/NonMaxSupress.h"

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
                    vector<Mat> &samples,       /* out: target objects, flipped( only for positive)*/
                    vector<Mat> &origsamples)   /* out: original target */
{
	int Nthreads = omp_get_max_threads();

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
        bf::directory_iterator end_it;
        vector<string> image_path_vector;
        vector<string> gt_path_vector;

        for( bf::directory_iterator file_iter(pos_img_path); file_iter!=end_it; file_iter++)
        {
            bf::path s = *(file_iter);
            string basename = bf::basename( s );
            string pathname = file_iter->path().string();
            string extname  = bf::extension( s );

            image_path_vector.push_back( pathname );
            /* read the gt according to the image name */
            gt_path_vector.push_back(opts.posGtDir + basename + ".txt");
        }

        #pragma omp parallel for num_threads(Nthreads) /* openmp -->but no error check in runtime ... */
        for( int i=0;i<image_path_vector.size();i++)
        {
            Mat im = imread( image_path_vector[i]);
            /* can not return from openmp body !*/
            //if(im.empty())
            //{
            //    cout<<"can not read image file "<<image_path_vector[i]<<endl;
            //    return false;
            //}

            vector<Rect> target_rects;
            FileStorage fst( gt_path_vector[i], FileStorage::READ | FileStorage::FORMAT_XML);
            //if(!fst.isOpened())
            //{
            //    cout<<"can not read gt file "<<gt_path_vector[i]<<endl;
            //    return false;
            //}
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
                #pragma omp critical
                {
                    origsamples.push_back( target_obj );
                }
            }
        }

        /* sample target if n_target > number_to_sample */
        if( origsamples.size() > number_to_sample)
        {
            std::random_shuffle( origsamples.begin(), origsamples.end());
            origsamples.resize( number_to_sample);
        }
        
        samples.resize( origsamples.size()*2); int copy_offset = origsamples.size();
        for(int i = 0; i<origsamples.size(); i++ )
        {
            samples[i] = origsamples[i].clone();
            Mat flipped_target; cv::flip( origsamples[i], flipped_target, 1 );
            samples[i+copy_offset] = flipped_target;
        }

        
    }
    else /* for negative samples */
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
        bf::directory_iterator end_it;int number_target = 0;
        for( bf::directory_iterator file_iter(neg_img_path); file_iter!=end_it; file_iter++)
		{
            string pathname = file_iter->path().string();
			neg_paths.push_back( pathname );
		}

		std::random_shuffle( neg_paths.begin(), neg_paths.end() );
		
        #pragma omp parallel for num_threads(Nthreads) /* openmp -->but no error check in runtime ... */
		for( int c=0;c<neg_paths.size() ;c++)
		{
			vector<Rect> target_rects;
			Mat img = imread( neg_paths[c] );
            //if(img.empty())
            //{
            //    cout<<"can not read image "<<neg_paths[c]<<endl;
            //    return false;
            //}
			/*  inf stage == 0, first time just sample the image, otherwise add the "hard sample" */
			if( stage==0 )
			{
                /*  sampling and shuffle  */
				sampleRects( number_target_per_image, img.size(), opts.modelDs, target_rects );
                std::random_shuffle( target_rects.begin(), target_rects.end() );
			}
			else
			{
                /* boostrap the negative samples */
                vector<double> conf_v;
                sc.detectMultiScale( img, target_rects, conf_v );
                if( target_rects.size() > number_target_per_image )
                    target_rects.resize( number_target_per_image);
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

                #pragma omp critical
                {
                    origsamples.push_back( target_obj );
                }
            }
		}

        /* random shuffle and sampling  */
        if(origsamples.size() > number_to_sample)
            origsamples.resize( number_to_sample );
    }
}

void copySamples( const vector<Mat> &in_samples,
                  vector<Mat> &out_samples)
{
    out_samples.clear();
    out_samples.resize( in_samples.size());
    for ( int c=0;c<in_samples.size() ;c++ ) 
    {
        in_samples[c].copyTo(out_samples[c]);
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
    feature_Pyramids ff1;
    softcascade sc;

    tree_para tree_par;
	cascadeParameter cas_para;
    detector_opt det_opt;
 
    tree_par.nBins = 256;
    tree_par.maxDepth = 2;
    tree_par.fracFtrs = 0.0625;

    //cas_para.posGtDir  = "/mnt/disk1/data/INRIAPerson/Train/posGT/";
    cas_para.posGtDir  = "/media/yuanyang/disk1/libs/piotr_toolbox/data/Inria/train/posGt_opencv/";
    //cas_para.posImgDir = "/mnt/disk1/data/INRIAPerson/Train/pos"; 
    cas_para.posImgDir = "/media/yuanyang/disk1/libs/piotr_toolbox/data/Inria/train/pos/"; 
	//cas_para.negImgDir = "/mnt/disk1/data/INRIAPerson/Train/neg/";
	cas_para.negImgDir = "/media/yuanyang/disk1/libs/piotr_toolbox/data/Inria/train/neg/";
    cas_para.infos = "2015-1-14, YuanYang, Test";
    cas_para.shrink = det_opt.shrink;
    cas_para.nchannels = 10;

    sc.setParas( cas_para);
    sc.setDebug( false);
    sc.setFeatureGen( ff1 );

    //vector<Mat> neg_samples;
    //vector<Mat> neg_origsamples;
    //vector<Mat> neg_previousSamples;
    //cout<<"reading and sampling pos&neg image patches "<<endl;
    //tk.start();
    //sampleWins( sc, 0, false, neg_samples, neg_origsamples);
    //
    ////for ( int c=0;c<neg_origsamples.size();c++) 
    ////{
    ////    imshow("sample", neg_origsamples[c] );
    ////    imwrite("sample.png", neg_origsamples[c]);
    ////    waitKey(0);
    ////}
	//
    //vector<Mat> pos_samples;
    //vector<Mat> pos_origsamples;
    //sampleWins( sc, 0, true, pos_samples, pos_origsamples);
    //tk.stop();
	//cout<<"# sampling time consuming is "<<tk.getTimeSec()<<" seconds "<<endl;
	//cout<<"neg sample number -> "<<neg_origsamples.size()<<endl;
	//cout<<"pos sample number -> "<<pos_samples.size()<<endl;

    ///*  ------------------------- test Adaboost for one stage --------------------------- */
    //Size modelDsPad = cas_para.modelDsPad;
    //int n_channels = cas_para.nchannels;
    //int n_shrink = cas_para.shrink;
    //int final_feature_dim = modelDsPad.width/n_shrink*modelDsPad.height/n_shrink*n_channels;
    //Mat pos_train_data = Mat::zeros( final_feature_dim, pos_samples.size(), CV_32F);
    //Mat neg_train_data = Mat::zeros( final_feature_dim, neg_origsamples.size(), CV_32F);

    //tk.reset();tk.start();
    //cout<<"computing features for file "<<endl;

    //#pragma omp parallel for num_threads(Nthreads)
    //for ( int c=0;c<pos_origsamples.size();c++) 
    //{
    //    vector<Mat> feas;
    //    ff1.computeChannels( pos_origsamples[c], feas );
    //    Mat tmp = pos_train_data.col(c);
    //    makeTrainData( feas, tmp , cas_para.modelDsPad, cas_para.shrink);
    //    if(c==0)
    //    {
    //        cout<<"channels is "<<feas.size()<<endl;
    //        cout<<"feature dim col "<<feas[c].cols<<" "<<feas[c].rows<<" final size "<<final_feature_dim<<endl;
    //    }
    //}

    //#pragma omp parallel for num_threads(Nthreads)
    //for ( int c=0;c<neg_origsamples.size();c++)
    //{
    //    vector<Mat> feas;
    //    ff1.computeChannels( neg_origsamples[c], feas );
    //    Mat tmp = neg_train_data.col(c);
    //    makeTrainData( feas, tmp , cas_para.modelDsPad, cas_para.shrink);
    //}
    //tk.stop();
    //cout<<"#time consuming "<<tk.getTimeSec()<<" seconds, computing features done for "<<neg_origsamples.size() + pos_origsamples.size()<<" targets "<<endl;


    //Adaboost ab;ab.SetDebug(false);
    //tree_para ad_para;
    //ad_para.nBins = 256;
    //ad_para.maxDepth = 2;
    //ad_para.fracFtrs = 0.0625;

    ///* check adaboost training*/
    //ab.Train( neg_train_data, pos_train_data, 64, ad_para );
    //vector<Adaboost> v_ab;
    //v_ab.push_back( ab);
    //sc.Combine( v_ab );

    //
    ///*  save and load */
    //sc.Save( "test.xml" );
    //sc.Load( "test.xml");
    //
    ///*  check softcascade combining  */
    //pos_train_data = pos_train_data.t();
    //double h = 0;
    //double fn =0;
    //for ( int c=0;c<pos_train_data.rows;c++ ) 
    //{
    //    h = 0;
    //    float *pp = pos_train_data.ptr<float>(c);
	//	sc.Predict( pp, h );
	//	if( h < 0)
	//		fn++;
    //}
    //cout<<"fn is "<<fn/pos_train_data.rows<<endl;


    ///* test softcascade Apply function */
    //Mat test_apply = imread("/media/yuanyang/disk1/git/adaboost/build/softcascade/test_2.png");

    //vector<Mat> f_chns;
    //vector<Rect> rs;vector<double> conf;


    //vector< vector<Mat> > approPyramid;
    //vector<double> appro_scales;
    //ff1.chnsPyramid( test_apply, approPyramid, appro_scales);
    //
    ////for ( int c=0; c<appro_scales.size();c++ ) 
    ////{
    ////    cout<<"scale "<<c<<" is "<<appro_scales[c]<<endl;
    ////}

    //tk.reset();tk.start();
    ////sc.Apply( test_apply, rs, conf);
    //sc.detectMultiScale( test_apply, rs, conf );
    //tk.stop();

    //cout<<"# time computing featuers and running the classifier is "<<tk.getTimeSec()<<endl;

    //cout<<"det size "<<rs.size()<<endl;
    //Mat draw1  = test_apply.clone();
    //Mat draw2  = test_apply.clone();
    //for(int c=0;c<rs.size(); c++)
    //{
    //    rectangle( draw1, rs[c], Scalar(255,0,128));
    //}

    //imshow( "before", draw1);

    //NonMaxSupress( rs, conf );

    //for(int c=0;c<rs.size(); c++)
    //{
    //    cout<<"confidence "<<conf[c]<<endl;
    //    rectangle( draw2, rs[c], Scalar(255,0,128));
    //}
    //cout<<"remained "<<rs.size()<<endl;
    //imshow("after", draw2);
    //waitKey(0);

    //*/

    /*  ------------------------- test Adaboost for one stage --------------------------- */

    vector<Mat> neg_samples;
    vector<Mat> neg_origsamples;
    vector<Mat> neg_previousSamples;

    vector<Mat> pos_samples;
    vector<Mat> pos_origsamples;

    Size modelDsPad = cas_para.modelDsPad;
    int n_channels = cas_para.nchannels;
    int n_shrink = cas_para.shrink;
    int final_feature_dim = modelDsPad.width/n_shrink*modelDsPad.height/n_shrink*n_channels;

    Mat pos_train_data;
    Mat neg_train_data;

    vector<Adaboost> v_ab;
	/*-----------------------------------------------------------------------------
	 *  Step 2 : iterate bootstraping and training
	 *-----------------------------------------------------------------------------*/
	for( int stage=0;stage<cas_para.nWeaks.size();stage++)
	{
		tk.reset();tk.start();
		cout<<"=========== Training Stage No "<<stage<<" ==========="<<endl;
		/* 1--> sample positives and compute info about channels */
		if( stage == 0)
		{
            sampleWins( sc, stage, true, pos_samples, pos_origsamples);
		}

		/* TODO 2--> compute lambdas */

		/* 3-->  compute features for positives */
        if( stage == 0)
        {
            cout<<"->Making positive training data ";
            pos_train_data = Mat::zeros( final_feature_dim, pos_samples.size(), CV_32F);
            #pragma omp parallel for num_threads(Nthreads)
            for ( int c=0;c<pos_samples.size();c++) 
            {
                vector<Mat> feas;
                ff1.computeChannels( pos_samples[c], feas );
                Mat tmp = pos_train_data.col(c);
                makeTrainData( feas, tmp , cas_para.modelDsPad, cas_para.shrink);
            }
            /* delete others */
            vector<Mat>().swap(pos_samples);
            vector<Mat>().swap(pos_origsamples);
            cout<<"done. number :"<<pos_train_data.cols<<endl;
        }


		/* 4--> sample negatives and compute features, accumulate negatives from previous stages */
        cout<<"->Making negative training data ";
        sampleWins( sc, stage, false, neg_samples, neg_origsamples );          /* remember the neg_samples is empty */
        cout<<"->Sampling done"<<endl;

        vector<Mat> accu_neg;
        if( neg_previousSamples.size() ==0 )                                   /* stage == 0 */
        {
            accu_neg = neg_origsamples;
        }
        else
        {
            int n1 = std::max(cas_para.nAccNeg,cas_para.nNeg)-neg_origsamples.size();   /* how many will be save from previous stage */
            if( n1 > neg_previousSamples.size())
            {
                std::random_shuffle( neg_previousSamples.begin(), neg_previousSamples.end());
                neg_previousSamples.resize( n1 );
            }
            accu_neg.reserve( neg_previousSamples.size() + neg_origsamples.size() );
            accu_neg.insert( accu_neg.begin(), neg_previousSamples.begin(), neg_previousSamples.end());
            accu_neg.insert( accu_neg.begin(), neg_origsamples.begin(), neg_origsamples.end());    

        }
        neg_previousSamples = accu_neg;
        neg_train_data = Mat::zeros( final_feature_dim, accu_neg.size(), CV_32F);
        cout<<"Neg samples now "<<accu_neg.size()<<endl;

        #pragma omp parallel for num_threads(Nthreads)
        for ( int c=0;c<accu_neg.size();c++)
        {
            vector<Mat> feas;
            ff1.computeChannels( accu_neg[c], feas );
            Mat tmp = neg_train_data.col(c);
            makeTrainData( feas, tmp , cas_para.modelDsPad, cas_para.shrink);
        }
        cout<<"done. number : "<<accu_neg.size()<<endl;

		/* 5-->  train boosted classifiers */
        Adaboost ab;ab.SetDebug(false);  
        cout<<"-- Training with "<<cas_para.nWeaks[stage]<<" weak classifiers."<<endl;
        ab.Train( neg_train_data, pos_train_data, cas_para.nWeaks[stage], tree_par);
        
        v_ab.push_back( ab );
        sc.Combine( v_ab );


        vector<Rect> re;
        vector<double> confs;
        cout<<"little test ";
        Mat test_img = imread("guozu.png");
        sc.detectMultiScale( test_img, re, confs );
        for( int c=0;c<re.size();c++)
        {
            rectangle( test_img, re[c], Scalar(255,0,0) );
        }
        imshow("testing result ~~", test_img );
        waitKey(0);
		tk.stop();
		cout<<"Done Stage No "<<stage<<" , time "<<tk.getTimeSec()<<endl<<endl;
	}

}
