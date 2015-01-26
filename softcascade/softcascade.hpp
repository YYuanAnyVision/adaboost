#ifndef SOFTCASCADE
#define SOFTCASCADE

#include <string>
#include <iostream>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "../Adaboost/Adaboost.hpp"
#include "../binaryTree/binarytree.hpp"

#include "../chnfeature/Pyramid.h"

using namespace std;
using namespace cv;


struct cascadeParameter
{
	vector<int> filter;					/* de-correlation filter parameters, eg [5 5]  */
	Size modelDs;						/* model height+width without padding (eg [100 41]) */
	Size modelDsPad;					/* model height+width with padding (eg [128 64])*/
	

	int stride;							/* [4] spatial stride between detection windows */
	double cascThr;						/* [-1] constant cascade threshold (affects speed/accuracy)*/
	double cascCal;						/* [.005] cascade calibration (affects speed/accuracy) */
	vector<int> nWeaks;					/* [128] vector defining number weak clfs per stagemodel eg[64 128 256 1024]*/
	int pBoost_nweaks;					/* parameters for boosting */
	tree_para pBoost_pTree;				/* parameters for boosting */
	string infos;						/* other informations~~ */
	int nPos;							/* [-1 -> inf] max number of pos windows to sample */						
	int nNeg;							/* [5000] max number of neg windows to sample*/
	int nPerNeg;						/* [25]  max number of neg windows to sample per image*/
	int nAccNeg;						/* [10000] max number of neg windows to accumulate*/

    string posGtDir;                    /* positive groundtruth directory */
    string posImgDir;                   /* positive image directory */
    string negImgDir;                   /* negative image directory */

    Size pad;                           /* ----------> get it from feature generator */
    int    nchannels;                   /* ----------> number of channels, usually 1 */
	int shrink;							/* ----------> should be provided by the chnPyramid */

	cascadeParameter()
	{
		modelDs    = Size(41, 100);
		modelDsPad = Size(64, 128);
		stride     = 4;
		cascThr	   = -1;
		cascCal    = 0.001;
		nWeaks.push_back( 32);
		nWeaks.push_back( 128);
		nWeaks.push_back( 512);
		nWeaks.push_back( 2048);

		pBoost_nweaks = 128;
		pBoost_pTree = tree_para();
		infos = "no infos";
		nPos = -1;
		nNeg = 5000;
		nPerNeg = 25;
		nAccNeg = 10000;

		shrink = 4;

        posGtDir = "";
        posImgDir = "";
        negImgDir = "";
        nchannels = 10;
	}
};


class softcascade
{
	public:

		/* 
		 * ===  FUNCTION  ======================================================================
		 *         Name:  Load
		 *  Description:  load models from path
		 * =====================================================================================
		 */
		bool Load( string path_to_model );		/* in : path of the model, shoule be a xml file saved by opencv FileStorage */

		/* 
		 * ===  FUNCTION  ======================================================================
		 *         Name:  Apply
		 *  Description:  predict the result giving data
		 *           in:  input_data, column vectors, same type as training data
		 *          out:  detect result
		 * =====================================================================================
		 */
		bool Apply( const vector<Mat> &input_data,		/*  in: channel features, input_data.size() == nchannels */
				    vector<Rect> &results,              /* out: detect results */
                    vector<double> &confidence) const;	/* out: detect confidence */

        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  Apply overload
         *  Description:  same as above, only computes the feature inside
         * =====================================================================================
         */
        bool Apply( const Mat &input_image,             /*  in: !!! image !!! */
				    vector<Rect> &results,              /* out: detect results */
                    vector<double> &confidence);	/* out: detect confidence */
        
        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  detectMultiScale
         *  Description:  detect targets in a given image using sliding windows
         * =====================================================================================
         */
        bool detectMultiScale( const Mat &image,                    /* in : image */
                               vector<Rect> &targets,               /* out: target positions*/
                               vector<double> &confidence,          /* out: target confidence */
                               int stride = 4,                      /* in : detection stride */
                               int minSize = 32,                    /* in : min target size */
                               int maxSize = 300) const;            /* in : max target size */

		/* 
		 * ===  FUNCTION  ======================================================================
		 *         Name:  Predict
		 *  Description:  test a single sample
		 *			 in:  data, has to be continuous!!
		 *			out:  predicted score ( hs ) 
		 *			      !!! this function shoule not be used in sliding window search, since the
		 *			      memory is not continuous for each window Rect
		 * =====================================================================================
		 */
		template < typename T> bool Predict(  const T *data,            /* in : test data, must be continuous in memory */
                                              double &score) const      /* out: score ..*/
		{
			if(!checkModel())
				return false;
			double h = 0;
			if(m_tree_depth != 0)
			{
				for( int t=0;t<m_number_of_trees;t++)
				{
					int position = 0;
					
					const int *t_child   = m_child.ptr<int>(t);
					const int *t_fids    = m_fids.ptr<int>(t);
					const double *t_thrs = m_thrs.ptr<double>(t);
					const double *t_hs   = m_hs.ptr<double>(t);

					while( t_child[position])
					{
						position = (( data[t_fids[position]] < t_thrs[position]) ? position*2+1:position*2+2);
					}
					h += t_hs[position];
                    if( h < m_opts.cascThr)
                        break;
				}

			}
			else
			{
				for( int c=0;c<m_number_of_trees;c++)
				{
					int position   = 0;
					const int *t_child   = m_child.ptr<int>(c);
					const int *t_fids    = m_fids.ptr<int>(c);
					const double *t_thrs = m_thrs.ptr<double>(c);
					const double *t_hs   = m_hs.ptr<double>(c);
					while( t_child[position] )  /*  iterate the tree */
					{
						position = (( data[t_fids[position]] < t_thrs[position]) ? t_child[position]: t_child[position] + 1);
					}
					h += t_hs[position];
                    if( h < m_opts.cascThr)
                        break;
				}
			}
			score = h;
		}
		

        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  Predict
         *  Description:  output the confidence of the inputData matrix( column vector)
         * =====================================================================================
         */
        bool Predict( const Mat &testData,              /* in : featureDim x numberOfSamples */
                      Mat &confidence)                  /* out: numberOfSamples x 1 */
        {
            if( !testData.isContinuous())
            {
                cout<<"Data must be continuous is Predict() function"<<endl;
                return false;
            }
            if( testData.empty())
            {
                cout<<"Data's empty in Predict() function "<<endl;
                return false;
            }
            if( testData.channels() != 1)
            {
                cout<<"only single channel is supported in Predict() function "<<endl;
                return false;
            }
            Mat tmp;        /* copy each vector into it, since opencv matrix is store by rows, *column* is not continuous in memory(except single column) */
            confidence = Mat::zeros( testData.cols, 1, CV_64F );
            for( int c=0;c<testData.cols;c++)
            {
                Mat t = testData.col(c);
                t.copyTo( tmp );
                tmp = tmp.t();
                double score = 0;
                if( testData.type() == CV_32F)
                    Predict( (const float*)tmp.data, score );
                else if( testData.type() == CV_64F )
                    Predict( (const double*)tmp.data, score );
                else if( testData.type() == CV_32S )
                    Predict( (const int*)tmp.data, score );
                else
                {
                    cout<<"no known data type in Predict() function "<<endl;
                    return false;
                }
                confidence.at<double>(c,0) = score;
            }
        }

		/* 
		 * ===  FUNCTION  ======================================================================
		 *         Name:  Save
		 *  Description:  save model
		 * =====================================================================================
		 */
		bool Save( string path_to_model );		/*  in: where to save the model, models is saved by opencv FileStorage */

		
		/* 
		 * ===  FUNCTION  ======================================================================
		 *         Name:  Combine
		 *  Description:  combine the adaboost class to a long cascade
		 *				  adaboost  is trained as the stage, each has different number of trees, eg
		 *				  [ad1 ad2 ad3 ad4 ]  -> [32 64 256 1024], now make a long cascade contains 1376 trees
		 * =====================================================================================
		 */
		bool Combine( vector<Adaboost> &ads );

		/* 
		 * ===  FUNCTION  ======================================================================
		 *         Name:  checkModel
		 *  Description:  check if the model is loaded right
		 * =====================================================================================
		 */
		bool checkModel() const;

        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  setDebug
         *  Description:  wanna output
         * =====================================================================================
         */
        void setDebug( bool m_d );

		/* 
		 * ===  FUNCTION  ======================================================================
		 *         Name:  getParas
		 *  Description:  return the parameters
		 * =====================================================================================
		 */
		cascadeParameter getParas() const;

        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  setParas
         *  Description:  
         * =====================================================================================
         */
        void setParas( const cascadeParameter &in_par );

        /* 
         * ===  FUNCTION  ======================================================================
         *         Name:  setFeatureGen
         *  Description:  set the feature generator, feature_Pyramids has a simple data structure
         * =====================================================================================
         */
        void setFeatureGen( const feature_Pyramids &in_fea_gen )
        {
            m_feature_gen = in_fea_gen;
        }

	private:

		/* 
		 * ===  FUNCTION  ======================================================================
		 *         Name:  setTreeDepth
		 *  Description:  set the depth of all leaf nodes, 0 if leaf depth varies
		 * =====================================================================================
		 */
		bool setTreeDepth();


	private:
		Mat m_fids;							/* nxK 32S feature index for each node , n -> number of trees, K -> number of nodes*/
		Mat m_thrs;							/* nxK 64F thresholds for each node */
		Mat m_child;						/* nxK 32S child index for each node */
		Mat m_hs;							/* nxK 64F log ratio (.5*log(p/(1-p)) at each node  */
		Mat m_weights;						/* nxK 64F total sample weight at each node */
		Mat m_depth;						/* nxK 32S depth of node*/
		Mat m_nodes;						/* nx1 32S number of nodes of each tree */
		bool m_debug;						/* wanna output? */
		int m_number_of_trees;				/* . */
		int m_tree_nodes;					/* if all the tree have the same structure, this will be the number of the nodes of each tree, otherwise -1*/
		int m_tree_depth;					/* depth of all leaf nodes (or 0 if leaf depth varies) */

		cascadeParameter m_opts;            /* detectot options  */
        feature_Pyramids m_feature_gen;     /*  feature generator */
};
#endif
