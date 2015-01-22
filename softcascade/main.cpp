#include <iostream>
#include <vector>
#include <algorithm>
#include <sstream>
#include <ctime>

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

#define TEST_STAT

using namespace std;
using namespace cv;

namespace bf = boost::filesystem;
namespace bl = boost::lambda;

/*  random generator function */
int myrandom (int i) { return std::rand()%i;}

/* extract the center part of the feature( crossponding to the target) , save it as 
 * clomun in output_data(not continuous) */
void makeTrainData( vector<Mat> &in_data, Mat &output_data, Size modelDs, int shrink)
{
    assert( output_data.type() == CV_32F);
    assert( in_data[0].type() == CV_32F && in_data[0].isContinuous());

	int w_in_data = in_data[0].cols;
	int h_in_data = in_data[0].rows;

	int w_f = modelDs.width/shrink;
	int h_f = modelDs.height/shrink;
    
    assert( w_in_data > w_f && h_in_data > h_f );
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
    cout<<"Sampling ..."<<endl;
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
            std::random_shuffle( origsamples.begin(), origsamples.end(), myrandom);
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

		std::random_shuffle( neg_paths.begin(), neg_paths.end(),myrandom);
		
        double ratio_neg_image_to_sample = 0.33;
        int    shrink_neg_number  = (int) neg_paths.size()*ratio_neg_image_to_sample;
       
        #pragma omp parallel for num_threads(Nthreads) /* openmp -->but no error check in runtime ... */
		for( int c=0;c<shrink_neg_number;c++)
		{
			vector<Rect> target_rects;
            vector<double> conf_v;

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
                std::random_shuffle( target_rects.begin(), target_rects.end() , myrandom);
			}
			else
			{
                /* boostrap the negative samples */
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
        {
            std::random_shuffle( origsamples.begin(), origsamples.end(), myrandom);
            origsamples.resize( number_to_sample );
        }
    }
    cout<<"Sampling done "<<endl;
    return true;
}
    

/* detector parameter define */
int main( int argc, char** argv)
{
    std::srand ( unsigned ( std::time(0) ) );
    Mat Km = get_Km(1);

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
    tree_par.fracFtrs = 0.08;

    //cas_para.posGtDir  = "/mnt/disk1/data/INRIAPerson/Train/posGT/";
    cas_para.posGtDir  = "/media/yuanyang/disk1/libs/piotr_toolbox/data/Inria/train/posGt_opencv/";
    //cas_para.posImgDir = "/mnt/disk1/data/INRIAPerson/Train/pos"; 
    cas_para.posImgDir = "/media/yuanyang/disk1/libs/piotr_toolbox/data/Inria/train/pos/"; 
	//cas_para.negImgDir = "/mnt/disk1/data/INRIAPerson/Train/neg/";
	cas_para.negImgDir = "/media/yuanyang/disk1/libs/piotr_toolbox/data/Inria/train/neg/";
    cas_para.infos = "2015-1-22, YuanYang, Test";
    cas_para.shrink = det_opt.shrink;
    cas_para.nchannels = 10;

    sc.setParas( cas_para);
    sc.setDebug( false);
    sc.setFeatureGen( ff1 );

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
		/*  1--> sample positives and compute info about channels */
		/*  2--> compute lambdas */
		if( stage == 0)
		{
            sampleWins( sc, stage, true, pos_samples, pos_origsamples);
            ff1.compute_lambdas( pos_origsamples );
		}

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
                
                for(int i=0;i<feas.size();i++)
                {
                    ff1.convTri( feas[i], feas[i], Km );
                }

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
            vector<Mat>().swap(accu_neg);accu_neg.clear();                      
            int n1 = std::max(cas_para.nAccNeg,cas_para.nNeg)-neg_origsamples.size();   /* how many will be save from previous stage */
            if( n1 < neg_previousSamples.size())
            {
                std::random_shuffle( neg_previousSamples.begin(), neg_previousSamples.end(), myrandom);
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
            for(int i=0;i<feas.size();i++)
            {
                ff1.convTri( feas[i], feas[i], Km );
            }
            Mat tmp = neg_train_data.col(c);
            makeTrainData( feas, tmp , cas_para.modelDsPad, cas_para.shrink);
        }
        cout<<"done. number : "<<accu_neg.size()<<endl;

        cout<<"neg_train_data's size "<<neg_train_data.size()<<"feature dim "<<neg_train_data.rows<<endl;
        cout<<"pos_train_data's size "<<pos_train_data.size()<<"feature dim "<<pos_train_data.rows<<endl;


		/* 5-->  train boosted classifiers */
        Adaboost ab;ab.SetDebug(false);  
        cout<<"-- Training with "<<cas_para.nWeaks[stage]<<" weak classifiers."<<endl;
        ab.Train( neg_train_data, pos_train_data, cas_para.nWeaks[stage], tree_par);
        

        vector<Adaboost> t_v;
        t_v.push_back( ab );
        sc.Combine( t_v );
		cout<<"Done Stage No "<<stage<<" , time "<<tk.getTimeSec()<<endl<<endl;

        /* ---------- ~show improvement over diffierent stages~ ------------*/
        vector<Rect> re;vector<double> confs;
        Mat test_img = imread("crop001573.png");
        tk.reset();tk.start();
        sc.detectMultiScale( test_img, re, confs );
        tk.stop();
        cout<<"Time consuming for detect a size "<<test_img.size()<<" pic is "<<tk.getTimeSec()<<endl;
        for( int c=0;c<re.size();c++)
        {
            if( confs[c] < 1 )
                continue;
            cout<<"confidence is "<<confs[c]<<endl;
            rectangle( test_img, re[c], Scalar(255,0,0), 3);
        }
        stringstream ss;ss<<stage;string stage_index;ss>>stage_index;
		tk.stop();
        //imshow("show",test_img);
        //waitKey(0);

	}
    sc.Save( "sc.xml");
    /*  swap the Mat data */
    neg_train_data = Mat::zeros(1,1,CV_32F);
    pos_train_data = Mat::zeros(1,1,CV_32F);
    
    /*----------------   test detectMultiScale over dataset , show ----------------*/
#ifdef TEST_MUITI
    bf::directory_iterator end_it;
    bf::path test_data_path("/media/yuanyang/disk1/libs/piotr_toolbox/data/Inria/Test/pos/");
    for( bf::directory_iterator file_iter(test_data_path); file_iter!=end_it; file_iter++)
    {
        string pathname = file_iter->path().string();
        Mat test_img = imread( pathname );

        vector<Rect> re;vector<double> confs;
        sc.detectMultiScale( test_img, re, confs);
        for( int c=0;c<re.size();c++)
        {
            if( confs[c] < 0 )
                continue;
            cout<<"confidence is "<<confs[c]<<endl;
            rectangle( test_img, re[c], Scalar(255,0,0), 3);
        }
        cout<<endl;
        imshow("testimage", test_img );
        waitKey(0);
    }
#endif

#ifdef TEST_STAT
    string testset_neg_path = "/media/yuanyang/disk1/libs/piotr_toolbox/data/Inria/Test/neg/";
    string testset_pos_image_path = "/media/yuanyang/disk1/libs/piotr_toolbox/data/Inria/Test/pos/";
    string testset_pos_gt_path = "/media/yuanyang/disk1/libs/piotr_toolbox/data/Inria/Test/AnnotTest/";
    
    /*  using the sampleWins function */
    cascadeParameter par_for_test = sc.getParas();
    par_for_test.negImgDir = testset_neg_path;
    par_for_test.posGtDir  = testset_pos_gt_path;
    par_for_test.posImgDir = testset_pos_image_path;
    
    sc.setParas( par_for_test);
    vector<Mat> test_pos_orig;
    vector<Mat> test_pos_all;
    sampleWins( sc, 0, true, test_pos_all, test_pos_orig );

    cout<<"->Making positive test data ";
    Mat pos_test_data = Mat::zeros( final_feature_dim, test_pos_all.size(), CV_32F);
    #pragma omp parallel for num_threads(Nthreads)
    for ( int c=0;c<test_pos_all.size();c++) 
    {
        vector<Mat> feas;
        ff1.computeChannels( test_pos_all[c], feas );
        
        for(int i=0;i<feas.size();i++)
        {
            ff1.convTri( feas[i], feas[i], Km );
        }

        Mat tmp = pos_test_data.col(c);
        makeTrainData( feas, tmp , cas_para.modelDsPad, cas_para.shrink);
    }
    /* delete others */
    vector<Mat>().swap(test_pos_all);
    vector<Mat>().swap(test_pos_orig);
    cout<<"done. Total number :"<<pos_test_data.cols<<endl;
    
    Mat for_test_feat;
    double stat_fn = 0;
    double avg_pos_score = 0;
    for( int c=0;c<pos_test_data.cols;c++)
    {
        for_test_feat = pos_test_data.col(c);
        for_test_feat = for_test_feat.t();
        double score = 0;
        sc.Predict( (float*)for_test_feat.data, score );
        cout<<"Test positive sample is "<<score<<endl;
        avg_pos_score += score;
        if( score < 0)
            stat_fn += 1.0;
    }
    stat_fn = stat_fn / pos_test_data.cols;
    avg_pos_score /= pos_test_data.cols;

    vector<Mat> test_neg_orig;
    vector<Mat> test_neg_all;
    sampleWins( sc, 0, false, test_neg_all, test_neg_orig );
    cout<<"Making negative test data "<<endl;
    Mat neg_test_data = Mat::zeros( final_feature_dim, test_neg_orig.size(), CV_32F );
 
    #pragma omp parallel for num_threads(Nthreads)
    for ( int c=0;c<test_neg_orig.size();c++) 
    {
        vector<Mat> feas;
        ff1.computeChannels( test_neg_orig[c], feas );
        
        for(int i=0;i<feas.size();i++)
        {
            ff1.convTri( feas[i], feas[i], Km );
        }

        Mat tmp = neg_test_data.col(c);
        makeTrainData( feas, tmp , cas_para.modelDsPad, cas_para.shrink);
    }
    /* delete others */
    vector<Mat>().swap(test_neg_all);
    vector<Mat>().swap(test_neg_orig);
    cout<<"done. Total number :"<<neg_test_data.cols<<endl;
    
    double stat_fp = 0;
    double avg_neg_score = 0;
    for( int c=0; c<neg_test_data.cols;c++)
    {
        for_test_feat = neg_test_data.col(c);
        for_test_feat = for_test_feat;
        double score = 0;
        sc.Predict( (float*)for_test_feat.data, score);
        avg_neg_score += score;
        if( score > 0 )
            stat_fp += 1.0;
        cout<<"Test negative sample is "<<score<<endl;
    }
    stat_fp /= neg_test_data.cols;
    avg_neg_score /= neg_test_data.cols;

    cout<<"Test result on INRIA dataset\n FP is "<<stat_fp<<" FN is "<<stat_fn<<endl;
    cout<<"avg pos score is "<<avg_pos_score<<" avg neg score is "<<avg_neg_score<<endl;
#endif



    return 0;
}
