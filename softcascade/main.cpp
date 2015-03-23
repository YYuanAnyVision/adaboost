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

using namespace std;
using namespace cv;

namespace bf = boost::filesystem;
namespace bl = boost::lambda;

bool isSameTarget( Rect r1, Rect r2)
{
	Rect intersect = r1 & r2;
	if(intersect.width * intersect.height < 1)
		return false;
	
	double union_area = r1.width*r1.height + r2.width*r2.height - intersect.width*intersect.height;

	if( intersect.width*intersect.height/union_area < 0.5 )
		return false;

	return true;
}

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
                    bool hasGroundTruth,        /*  in: hasGroundTruth = false -> no need for xml, positive samples are cropped */
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
    
    if(isPositive)	//sample Positive samples or mining negative samples from positive..
    {
        if( hasGroundTruth)
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
                
                if( extname!=".jpg" && extname!=".bmp" && extname!=".png" &&
                        extname!=".JPG" && extname!=".BMP" && extname!=".PNG")
                    continue;

                /* check if both groundTruth and image exist */
                bf::path gt_path( opts.posGtDir + basename + ".xml");
                if(!bf::exists( gt_path))   // image already exists ..
                {
                    continue;
                }

                image_path_vector.push_back( pathname );
                /* read the gt according to the image name */
                gt_path_vector.push_back(opts.posGtDir + basename + ".xml");
            }
            

            #pragma omp parallel for num_threads(Nthreads) /* openmp -->but no error check in runtime ... */
            for( int i=0;i<image_path_vector.size();i++)
            {
                Mat im = imread( image_path_vector[i]);

                vector<Rect> target_rects;
                FileStorage fst( gt_path_vector[i], FileStorage::READ | FileStorage::FORMAT_XML);
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
        }
        else
        {
            bf::path pos_img_path( opts.posImgDir );

            if( !bf::exists( pos_img_path) )
            {
                cout<<"pos img or gt path does not exist!"<<endl;
                return false;
            }
            int number_pos_img = getNumberOfFilesInDir( opts.posGtDir );

            /* iterate the folder*/
            bf::directory_iterator end_it;
            vector<string> image_path_vector;
            for( bf::directory_iterator file_iter(pos_img_path); file_iter!=end_it; file_iter++)
            {
                bf::path s = *(file_iter);
                string basename = bf::basename( s );
                string pathname = file_iter->path().string();
                string extname  = bf::extension( s );
                
                if( extname!=".jpg" && extname!=".bmp" && extname!=".png" &&
                        extname!=".JPG" && extname!=".BMP" && extname!=".PNG")
                    continue;
                image_path_vector.push_back( pathname );
            }
            

            #pragma omp parallel for num_threads(Nthreads) /* openmp -->but no error check in runtime ... */
            for( int i=0;i<image_path_vector.size();i++)
            {
                Mat im = imread( image_path_vector[i]);

                /*  resize the rect to fixed widht / height ratio, for pedestrain det , is 41/100 for INRIA database */
                Rect target_rects = resizeToFixedRatio( target_rects, opts.modelDs.width*1.0/opts.modelDs.height, 1); /* respect to height */
                target_rects = Rect(0, 0, im.cols, im.rows);
                /* grow it a little bit */
                int modelDsBig_width = std::max( 8*opts.shrink, opts.modelDsPad.width)+std::max(2, 64/opts.shrink)*opts.shrink;
                int modelDsBig_height = std::max( 8*opts.shrink,opts.modelDsPad.height)+std::max(2,64/opts.shrink)*opts.shrink;
                double w_ratio = modelDsBig_width*1.0/opts.modelDs.width;
                double h_ratio = modelDsBig_height*1.0/opts.modelDs.height;
                target_rects = resizeBbox( target_rects, h_ratio, w_ratio);
                
                /* finally crop the image */
                Mat target_obj = cropImage( im, target_rects);
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
			string extname  = bf::extension( *file_iter);
			if( extname!=".jpg" && extname!=".bmp" && extname!=".png" &&
					extname!=".JPG" && extname!=".BMP" && extname!=".PNG")
				continue;
			neg_paths.push_back( pathname );
		}
		std::random_shuffle( neg_paths.begin(), neg_paths.end(),myrandom);
       
		#pragma omp parallel for num_threads(Nthreads) /* openmp -->but no error check in runtime ... */
		for( int c=0;c<number_of_neg_images ;c++)
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
				if( target_rects.size() > number_target_per_image)
					target_rects.resize( number_target_per_image );
			}
			else
			{
                /* boostrap the negative samples */
                sc.detectMultiScale( img, target_rects, conf_v, Size(0,0), Size(0,0) );

                if( target_rects.size() > number_target_per_image )
                    target_rects.resize( number_target_per_image);
			}
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
 
int runTrainAndTest()
{
    std::srand ( unsigned ( std::time(0) ) );
    Mat Km = get_Km(1);

    /* globale paras*/
	int Nthreads = omp_get_max_threads();
    cout<<"using threads "<<Nthreads<<endl;
    TickMeter tk;

	/*-----------------------------------------------------------------------------
	 *  Step 1 : set and check parameters
	 *-----------------------------------------------------------------------------*/
    feature_Pyramids ff1;
    softcascade sc;

    tree_para tree_par;
	cascadeParameter cas_para;
    channels_opt det_opt;
 
    tree_par.nBins = 256;
    tree_par.maxDepth = 2;
    tree_par.fracFtrs = 0.0625;

    bool has_groundtruth = true;

    cas_para.posGtDir  = "/home/yuanyang/Workspace/INRIA/train/posGt_opencv/";
    cas_para.posImgDir = "/home/yuanyang/Workspace/INRIA/train/pos/"; 
	cas_para.negImgDir = "/home/yuanyang/Workspace/INRIA/train/neg/";

    cas_para.infos = "2015-1-25, YuanYang, Test";
    cas_para.shrink = det_opt.shrink;
    cas_para.pad   = det_opt.pad;
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
		cout<<"=========== Training Stage No "<<stage<<" ==========="<<endl;
		/*  1--> sample positives and compute info about channels */
		/*  2--> compute lambdas */
		if( stage == 0)
		{
            sampleWins( sc, stage, true,has_groundtruth,  pos_samples, pos_origsamples);
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
                ff1.computeChannels_sse( pos_samples[c], feas );
                
                for(int i=0;i<feas.size();i++)
                {
                    ff1.convTri( feas[i], feas[i], 1,1);
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
        sampleWins( sc, stage, false, has_groundtruth,  neg_samples, neg_origsamples );          /* remember the neg_samples is empty */

        vector<Mat> accu_neg;
        if( stage ==0 )                                   /* stage == 0 */
        {
            accu_neg = neg_origsamples;
        }
        else
        {
            int n1 = std::max(cas_para.nAccNeg,cas_para.nNeg)-neg_origsamples.size();   /* how many will be save from previous stage */
			cout<<"add 'hard example' "<<neg_origsamples.size()<<", keep "<<n1<<" neg examples from previous stage "<<endl;
            if( n1 < neg_previousSamples.size())
            {
                std::random_shuffle( neg_previousSamples.begin(), neg_previousSamples.end(), myrandom);
                neg_previousSamples.resize( n1 );
            }
            accu_neg.reserve( neg_previousSamples.size() + neg_origsamples.size() );
            accu_neg.insert( accu_neg.begin(), neg_origsamples.begin(), neg_origsamples.end());    
            accu_neg.insert( accu_neg.begin(), neg_previousSamples.begin(), neg_previousSamples.end());
        }
        neg_previousSamples = accu_neg;
        neg_train_data = Mat::zeros( final_feature_dim, accu_neg.size(), CV_32F);

        cout<<"->Making negative training data ";
        #pragma omp parallel for num_threads(Nthreads)
        for ( int c=0;c<accu_neg.size();c++)
        {
            vector<Mat> feas;
            ff1.computeChannels_sse( accu_neg[c], feas );
            for(int i=0;i<feas.size();i++)
            {
                ff1.convTri( feas[i], feas[i], 1, 1);
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

        /* ------------- show the average pos and neg distance ------------- */
        double avg_train_pos_score = 0;
        double avg_train_neg_score = 0;
        Mat pre_neg_score;
        Mat pre_pos_score;
        ab.Apply( pos_train_data, pre_pos_score);
        ab.Apply( neg_train_data, pre_neg_score);
        for( int c=0;c<pre_pos_score.rows;c++)
        {
            avg_train_pos_score += pre_pos_score.at<double>(c,0);
        }
        avg_train_pos_score /= pre_pos_score.rows;
        for( int c=0;c<pre_neg_score.rows;c++)
        {
            avg_train_neg_score += pre_neg_score.at<double>(c,0);
        }
        avg_train_neg_score /= pre_neg_score.rows;
        cout<<"Train : avg_train_pos_score is "<<avg_train_pos_score<<endl;
        cout<<"Train : avg_train_neg_score is "<<avg_train_neg_score<<endl;
	}
    /*  swap the Mat data */
    neg_train_data = Mat::zeros(1,1,CV_32F);
    pos_train_data = Mat::zeros(1,1,CV_32F);

    /*  save the model */
    sc.Save("for_test_sc.xml");
    return 0;
}



/* detector parameter define */
int main( int argc, char** argv)
{
    return runTrainAndTest();
}
