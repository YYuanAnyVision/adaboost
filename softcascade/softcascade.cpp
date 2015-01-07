#include <iostream>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <cmath>
#include "opencv2/highgui/highgui.hpp"
#include "softcascade.hpp"
#include "../binaryTree/binarytree.hpp"
#include "../Adaboost/Adaboost.hpp"

using namespace cv;
using namespace std;

bool softcascade::Combine(vector<Adaboost> &ads )
{
	/*  ==================== get informations ==================== */
	/* target 1 -> calculate the total number of trees
	 * target 2 -> calculate max_number_of_nodes*/
	/* target 3 -> do they have the same tree  */

	/*  the final structure will be number_of_trees x max_number_of_nodes for fids, thrs, child etc */
	int number_of_trees = 0;
	int max_number_of_nodes = 0;
	int tree_nodes = 0;

	for ( int c=0;c<ads.size() ;c++ ) 
	{
		/*  target 3 */
		if( c == 0)
		{
			tree_nodes = ads[c].getTreesNodes();
		}
		else
		{
			if( tree_nodes == -1 || ads[c].getTreesNodes() == -1 || (tree_nodes != ads[c].getTreesNodes()) )
			{
				tree_nodes = -1;
			}
		}
		
		/*  target 1 */
		const vector<binaryTree> bts = ads[c].getTrees();
		if( bts.empty() )
		{
			cout<<"<softcascade::Combine><error> Adaboost struct is empty .. "<<endl;
			return false;
		}
		else
		{
			number_of_trees += bts.size();
		}

		/* target 2 */
		max_number_of_nodes = std::max( max_number_of_nodes, ads[c].getMaxNumNodes() );
	}


	/* ==========================   copy the tree data   ========================= */
	if(m_debug)
		cout<<"Making model from adaboost structure ..."<<endl;

	assert( number_of_trees > 0 && max_number_of_nodes > 0);
	m_fids    =  Mat::zeros( number_of_trees,max_number_of_nodes, CV_32S);
	m_child   =  Mat::zeros( number_of_trees,max_number_of_nodes, CV_32S);
	m_depth   =  Mat::zeros( number_of_trees,max_number_of_nodes, CV_32S);
	m_thrs    =  Mat::zeros( number_of_trees,max_number_of_nodes, CV_64F);
	m_hs      =  Mat::zeros( number_of_trees,max_number_of_nodes, CV_64F);
	m_weights =  Mat::zeros( number_of_trees,max_number_of_nodes, CV_64F);
	m_nodes   =  Mat::zeros( number_of_trees, 1, CV_32S);
	m_number_of_trees = number_of_trees;

	int counter = 0;
	for ( int c=0;c<ads.size();c++) 
	{
		Mat nodes_info = ads[c].getNodes();
		const vector<binaryTree> bts = ads[c].getTrees();
		for( int i=0;i<bts.size();i++)
		{
			const biTree *ptr = bts[i].getTree();		
			(*ptr).fids.copyTo( m_fids.row(counter).colRange(0,nodes_info.at<int>(i,0)) );
			(*ptr).child.copyTo( m_child.row(counter).colRange(0,nodes_info.at<int>(i,0)) );
			(*ptr).depth.copyTo( m_depth.row(counter).colRange(0,nodes_info.at<int>(i,0)) );
			(*ptr).hs.copyTo( m_hs.row(counter).colRange(0,nodes_info.at<int>(i,0)) );
			(*ptr).thrs.copyTo( m_thrs.row(counter).colRange(0,nodes_info.at<int>(i,0)) );
			(*ptr).weights.copyTo( m_weights.row(counter).colRange(0,nodes_info.at<int>(i,0)) );
			m_nodes.at<int>(counter,0) = nodes_info.at<int>(i,0);
			counter++;
		}
	}

	if(m_debug)
	{
		cout<<"fids samples \n"<<m_fids.rowRange(0,10)<<endl;
		cout<<"fids samples \n"<<m_fids.rowRange(number_of_trees-10,number_of_trees)<<endl;
		cout<<"depth samples \n"<<m_depth.rowRange(0,10)<<endl;
		cout<<"depth samples \n"<<m_depth.rowRange(number_of_trees-10,number_of_trees)<<endl;
		cout<<"hs samples \n"<<m_hs.rowRange(0,10)<<endl;
		cout<<"hs samples \n"<<m_hs.rowRange(number_of_trees-10,number_of_trees)<<endl;
		cout<<"thrs samples \n"<<m_thrs.rowRange(0,10)<<endl;
		cout<<"thrs samples \n"<<m_thrs.rowRange(number_of_trees-10,number_of_trees)<<endl;
		cout<<"weights samples \n"<<m_weights.rowRange(0,10)<<endl;
		cout<<"weights samples \n"<<m_weights.rowRange(number_of_trees-10 ,number_of_trees)<<endl;
		cout<<"child samples \n"<<m_child.rowRange(0,10)<<endl;
		cout<<"child samples \n"<<m_child.rowRange(number_of_trees-10, number_of_trees)<<endl;
		cout<<"nodes samples \n"<<m_nodes.rowRange(0,10)<<endl;
		cout<<"nodes samples \n"<<m_nodes.rowRange(number_of_trees-10,number_of_trees)<<endl;
		cout<<"tree_nodes is \n"<<tree_nodes<<endl;
		cout<<"number_of_trees is \n"<<number_of_trees<<endl;
		cout<<"max_number_of_nodes is \n"<<max_number_of_nodes<<endl;
	}
	return true;
}



bool softcascade::checkModel() const
{
	if( m_fids.empty() || m_thrs.empty() || m_child.empty() || m_weights.empty()|| m_hs.empty() || m_depth.empty())
	{
		cout<<"<softcascade::checkModel><error> Model is empty "<<endl;
		return false;
	}
	if( !m_fids.isContinuous() || !m_thrs.isContinuous() || !m_child.isContinuous() ||
			!m_depth.isContinuous() || !m_hs.isContinuous() || !m_weights.isContinuous())
	{
		cout<<"<softcascade::checkModel><error> Model is not continuous "<<endl;
		return false;
	}
	return true;
}

bool softcascade::Apply( const Mat &input_data,			/*  in: featuredim x number_of_samples */
						 vector<Rect> &results ) const	/* out: results */
{
	if(!checkModel())
		return false;
	if(!input_data.isContinuous())
	{
		cout<<"<softcascade::Apply><error> input_data shoule be continuous ~"<<endl;
		return false;
	}
}

template <typename T> void _apply( const T *input_data,		        	/* in : (nchannels*nheight)x(nwidth) channels feature */
								   const int in_width,		        	/* in : width of a single channel image */
								   const int in_height,		        	/* in : height of a single channel image */
								   const int nchannels,		        	/* in : number of the channel */
								   const Mat &fids,			        	/* in : (number_of_trees)x(number_of_nodes) fids matrix */
								   const Mat &child,		        	/* in : child .. same */
								   const Mat &thrs,						/* in : thrs */
								   const Mat &hs,						/* in : hs  */
								   const cascadeParameter &opts,		/* in : detector options */
								   vector<Rect> results)				/* out: detected results */
{
	const int shrink = opts.shrink;
	const int modelHeight = opts.modelDsPad.height;
	const int modelWidth  = opts.modelDsPad.width;
	const int stride      = opts.stride;
	const double cascThr  = opts.cascThr;
	const double cascCal  = opts.cascCal;
	
	const int number_of_trees = fids.rows;
	const int number_of_nodes = fids.cols;

	/* calculate the scan step on rows and cols */
	int n_height = (int)ceil((in_height*shrink-modelHeight+1)/stride);
	int n_width  = (int)ceil((in_width*shrink-modelWidth+1)stride);
	
	/* generate the feature index array */
	int total_dim = modelHeight/shrink*modelWidth/shrink*nchannels;
	unsigned int *cids = new unsigned int[total_dim];
	int counter=0;
	
	for( int c=0;c<nchannels;c++)
		for(int h=0;h<modelHeight/shrink;h++)
			for(int w=0;w<modelWidth/shrink;w++)
				cids[counter++] = c*in_width*in_height+in_width*h + w;

	/*  apply classifier to each block */
	for(int h=0;h<n_height;h++)
	{
		for(int w=0;w<n_width;w++)
		{
			T *probe_feature_starter = input_data + (h*stride/shrink)*in_width + (w*stride/shrink);
		}
	}
}


