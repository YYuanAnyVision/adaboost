#include <iostream>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <sstream>
#include "opencv2/highgui/highgui.hpp"
#include "softcascade.hpp"
#include "../binaryTree/binarytree.hpp"
#include "../Adaboost/Adaboost.hpp"
#include "../misc/NonMaxSupress.h"
#include "../misc/misc.hpp"

using namespace cv;
using namespace std;


template <typename T> void _apply( const T *input_data,                 /* in : (nchannels*nheight)x(nwidth) channels feature, already been scaled with shrink*/
                                   const int &in_width,                 /* in : width of a single channel image */
                                   const int &in_height,                /* in : height of a single channel image */
                                   const Mat &fids,                     /* in : (number_of_trees)x(number_of_nodes) fids matrix */
                                   const Mat &child,                    /* in : child .. same */
                                   const Mat &thrs,                     /* in : thrs */
                                   const Mat &hs,                       /* in : hs  */
                                   const cascadeParameter &opts,        /* in : detector options */
                                   const int &tree_depth,               /* in : 0 if tree varies, number of nodes otherwise */
                                   vector<Rect> &results,               /* out: detected results */
                                   vector<double> &confidence )         /* out: detection confidence, same size as results */
{
    
    const int shrink      = opts.shrink;
    const int modelHeight = opts.modelDsPad.height;
    const int modelWidth  = opts.modelDsPad.width;
    const int modelW_fit  = opts.modelDs.width;
    const int modelH_fit  = opts.modelDs.height;
    const int modelW_shift= (modelWidth - opts.modelDs.width)/2 - opts.pad.width;
    const int modelH_shift= (modelHeight- opts.modelDs.height)/2 - opts.pad.height;
    const int stride      = opts.stride;
    const double cascThr  = opts.cascThr;
    const double cascCal  = opts.cascCal;
    const int nchannels   = opts.nchannels;

    
    const int number_of_trees = fids.rows;
    const int number_of_nodes = fids.cols;

    /* calculate the scan step on rows and cols */
    int n_height = (int) ceil((in_height*shrink-modelHeight+1)/stride);
    int n_width  = (int) ceil((in_width*shrink-modelWidth+1)/stride);

    /* generate the feature index array */
    int total_dim = modelHeight/shrink*modelWidth/shrink*nchannels;
    unsigned int *cids = new unsigned int[total_dim];
    int counter=0;
    
    for( int c=0;c<nchannels;c++)
        for(int h=0;h<modelHeight/shrink;h++)
            for(int w=0;w<modelWidth/shrink;w++)
                cids[counter++] = c*in_width*in_height+in_width*h + w;

    /*  apply classifier to each block */
    for(int c=0;c<n_height;c++)
    {
        for(int w=0;w<n_width;w++)
        {
            const T *probe_feature_starter = input_data + (c*stride/shrink)*in_width + (w*stride/shrink);
            double h=0;
            
            /*  full tree case, save the look up operation with t_child*/
            if( tree_depth != 0)
            {
                for( int t=0;t<number_of_trees;t++)
                {
                    int position = 0;
                    
                    const int *t_child   = child.ptr<int>(t);
                    const int *t_fids    = fids.ptr<int>(t);
                    const double *t_thrs = thrs.ptr<double>(t);
                    const double *t_hs   = hs.ptr<double>(t);

                    while( t_child[position])
                    {
                        position = (( probe_feature_starter[cids[t_fids[position]]] < t_thrs[position]) ? position*2+1 : position*2+2);
                    }
                    h += t_hs[position];
                    if( h <=opts.cascThr)       /* reject once the score is less than cascade threshold */
                        break;
                }
            }
            else
            {
                for( int t=0;t<number_of_trees;t++)
                {
                    /*  using ptr() here is very efficient as opencv suggested, changing it to pure pointer operation
                     *  gains little */
                    int position = 0;
                    
                    const int *t_child   = child.ptr<int>(t);
                    const int *t_fids    = fids.ptr<int>(t);
                    const double *t_thrs = thrs.ptr<double>(t);
                    const double *t_hs   = hs.ptr<double>(t);
                    while( t_child[position])
                    {
                        position = (( probe_feature_starter[cids[t_fids[position]]] < t_thrs[position]) ? t_child[position]: t_child[position] + 1);
                    }
                    h += t_hs[position];
                    if( h <=opts.cascThr)       /* reject once the score is less than cascade threshold */
                        break;
                }
            }
            /* add detection result */
            if( h>opts.cascThr)
            {
                Rect tmp( w*stride+modelW_shift, c*stride+modelH_shift, modelW_fit, modelH_fit );
                results.push_back( tmp );
                confidence.push_back( h );
            }
        }
    }
    delete [] cids;
}
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
    m_tree_nodes = tree_nodes;

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

    /*  shift the hs */
    m_hs = m_hs + m_opts.cascCal;
    cout<<"softcascade : number of trees "<<m_number_of_trees<<endl;

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

    /*  set the depth */
    setTreeDepth();
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
        cout<<"the depth of the trees is "<<m_tree_depth<<endl;
    }
    return true;
}


bool softcascade::setTreeDepth()
{
    if(!checkModel())
        return false;

    m_tree_depth = 0;
    if( m_tree_nodes < 0 )
    {
        m_tree_depth = 0;
        return true;
    }

    /* all the trees have the same number of nodes, the last element should be leaf */
    int depppth = m_depth.at<int>(0,m_tree_nodes-1);
    for ( int i=0;i<m_number_of_trees ;i++ ) 
    {
        const int *t_child = m_child.ptr<int>(i);
        const int *t_depth = m_depth.ptr<int>(i);
        for ( int j=0;j<m_tree_nodes;j++ ) 
        {
            if( t_child[j] > 0 )
                continue;
            else
            {
                if( t_depth[j] == depppth)  
                    continue;
                else
                {
                    depppth = 0;
                    break;
                }
            }
        }
        if(depppth == 0)
            break;
    }
    m_tree_depth = depppth;
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

bool softcascade::Apply( const vector<Mat> &input_data,      /*  in: channels feature which has a continuous mem like nchannelsxfeature_widthxfeature_height*/
                         vector<Rect> &results,              /* out: results */ 
                         vector<double> &confidence) const   /* out: confidence */
{
    /*  --------------------------- check --------------------------------*/
    if(!checkModel())
        return false;

    if(!input_data[0].isContinuous())
    {
        cout<<"<softcascade::Apply><error> input_data shoule be continuous ~"<<endl;
        return false;
    }

    if( m_opts.nchannels != input_data.size())
    {
        cout<<"<softcascade::Apply><error> input_data's size should equ nchannels "<<endl;
        return false;
    }

    /* the nchannel features shoule be continuous in memory, check the pointer, for various type*/
    if( input_data[0].type() == CV_32F)
    {
        if( (const float*)(input_data[0].data) + (m_opts.nchannels-1)*input_data[0].cols*input_data[0].rows != (const float*)input_data[m_opts.nchannels-1].data )
        {
            cout<<"<softcascade::Apply><error> input_data's memory not continuous "<<endl;
            return false;
        }
        _apply( (const float*)input_data[0].data,input_data[0].cols,input_data[0].rows,m_fids,m_child,m_thrs,m_hs,m_opts,m_tree_depth,results,confidence);
    }
    else if(input_data[0].type() == CV_64F)
    {
        if( (const double*)(input_data[0].data)+(m_opts.nchannels-1)*input_data[0].cols*input_data[0].rows!=(const double*)input_data[m_opts.nchannels-1].data )
        {
            cout<<"<softcascade::Apply><error> input_data's memory not continuous "<<endl;
            return false;
        }
        _apply( (const double*)input_data[0].data,input_data[0].cols,input_data[0].rows,m_fids,m_child,m_thrs,m_hs,m_opts,m_tree_depth,results,confidence);
    }
    else if(input_data[0].type() == CV_32S)
    {
        if( (const int*)(input_data[0].data)+(m_opts.nchannels-1)*input_data[0].cols*input_data[0].rows!=(const int*)input_data[m_opts.nchannels-1].data )
        {
            cout<<"<softcascade::Apply><error> input_data's memory not continuous "<<endl;
            return false;
        }
        
        _apply( (const int*)input_data[0].data, input_data[0].cols, input_data[0].rows, m_fids, m_child, m_thrs, m_hs, m_opts, m_tree_depth, results, confidence);
    }
    else
    {
        cout<<"softcascade::Apply><error> unsupported data type, must be one of CV_64F, CV_32F or CV_32S "<<endl;
        return false;
    }
    /*  --------------------------- check done ----------------------------*/

}



bool softcascade::Save( string path_to_model )      /*  in: where to save the model, models is saved by opencv FileStorage */
{
    FileStorage fs( path_to_model, FileStorage::WRITE);
    if( !fs.isOpened())
    {
        cout<<"<softcascade::Save><error> can not open file "<<path_to_model<<" for writing ."<<endl;
        return false;
    }

    fs<<"m_fids"<<m_fids;
    fs<<"m_child"<<m_child;
    fs<<"m_hs"<<m_hs;
    fs<<"m_thrs"<<m_thrs;
    fs<<"m_weights"<<m_weights;
    fs<<"m_depth"<<m_depth;
    fs<<"m_nodes"<<m_nodes;
    fs<<"m_number_of_trees"<<m_number_of_trees;
    fs<<"m_tree_nodes"<<m_tree_nodes;
    
    /* write options .. */
    fs<<"m_opts_filter"<<"[";
    for(int c=0;c<m_opts.filter.size();c++)
        fs<<m_opts.filter[c];
    fs<<"]";

    fs<<"m_opts_modelDs"<<m_opts.modelDs;
    fs<<"m_pots_modelDsPad"<<m_opts.modelDsPad;
    fs<<"m_opts_shrink"<<m_opts.shrink;
    fs<<"m_opts_stride"<<m_opts.stride;
    fs<<"m_opts_nWeaks"<<"[";
    for( int c=0;c<m_opts.nWeaks.size();c++)
    {
        fs<<m_opts.nWeaks[c];
    }
    fs<<"]";
    fs<<"m_opts_cascThr"<<m_opts.cascThr;
    fs<<"m_opts_cascCal"<<m_opts.cascCal;
    fs<<"m_opts_pBoost_nweaks"<<m_opts.pBoost_nweaks;
    fs<<"m_opts_infos"<<m_opts.infos;
    fs<<"m_opts_nPos"<<m_opts.nPos;
    fs<<"m_opts_nNeg"<<m_opts.nNeg;
    fs<<"m_opts_nPerNeg"<<m_opts.nPerNeg;
    fs<<"m_opts_nAccNeg"<<m_opts.nAccNeg;
    fs.release();
    cout<<"Saving Model Done "<<endl;
    return true;
}

bool softcascade::Load( string path_to_model )      /* in : path of the model, shoule be a xml file saved by opencv FileStorage */
{
    FileStorage fs(path_to_model, FileStorage::READ );
    if(!fs.isOpened())
    {
        cout<<"<softcascade::Load><error> Can not load model file "<<path_to_model<<endl;
        return false;
    }
    
    fs["m_fids"]>>m_fids;
    fs["m_hs"]>>m_hs;
    fs["m_thrs"]>>m_thrs;
    fs["m_child"]>>m_child;
    fs["m_weights"]>>m_weights;
    fs["m_depth"] >> m_depth;
    fs["m_nodes"]>>m_nodes;
    fs["m_number_of_trees"]>>m_number_of_trees;
    fs["m_tree_nodes"]>>m_tree_nodes;
    fs["m_opts_filter"] >> m_opts.filter;
    fs["m_opts_modelDs"]>>m_opts.modelDs;
    fs["m_pots_modelDsPad"]>>m_opts.modelDsPad;
    fs["m_opts_shrink"]>>m_opts.shrink;
    fs["m_opts_stride"]>>m_opts.stride;
    fs["m_opts_nWeaks"]>>m_opts.nWeaks;
    fs["m_opts_cascThr"]>>m_opts.cascThr;
    fs["m_opts_cascCal"]>>m_opts.cascCal;
    fs["m_opts_pBoost_nweaks"]>>m_opts.pBoost_nweaks;
    fs["m_opts_infos"]>>m_opts.infos;
    fs["m_opts_nPos"]>>m_opts.nPos;
    fs["m_opts_nNeg"]>>m_opts.nNeg;
    fs["m_opts_nPerNeg"]>>m_opts.nPerNeg;
    fs["m_opts_nAccNeg"]>>m_opts.nAccNeg;
    
    cout<<"Loading Model Done "<<endl;
    cout<<"# Model Info --> "<<m_opts.infos<<endl;
    return true;
}
cascadeParameter softcascade::getParas() const
{
    return m_opts;
}

void softcascade::setParas( const cascadeParameter &in_par )
{
    m_opts = in_par; 
}

bool softcascade::detectMultiScale( const Mat &image,
                       vector<Rect> &targets,
                       vector<double> &confidence,
                       int stride,
                       int minSize,
                       int maxSize) const
{
    
    vector< vector<Mat> > approPyramid;
    vector<double> appro_scales;
    vector<double> scale_w;
    vector<double> scale_h;

    m_feature_gen.chnsPyramid( image, approPyramid, appro_scales, scale_h, scale_w);


    for( int c=0;c<approPyramid.size();c++)
    {
        vector<Rect> t_tar;
        vector<double> t_conf;
        Apply( approPyramid[c], t_tar, t_conf);
        for ( int i=0;i<t_tar.size(); i++) 
        {
            
            Rect s( t_tar[i].x/appro_scales[c], t_tar[i].y/appro_scales[c], t_tar[i].width/scale_w[c], t_tar[i].height/scale_h[c]  );
            targets.push_back( s );
            confidence.push_back( t_conf[i]);
        }
    }
    
    /*  non max supression */
    NonMaxSupress( targets, confidence );

    return true;
}
bool softcascade::Apply( const Mat &input_image,        /*  in: !!! image !!! */
                    vector<Rect> &results,              /* out: detect results */
                    vector<double> &confidence) 	    /* out: detect confidence */
{
    vector<Mat> chns;
    m_feature_gen.computeChannels( input_image, chns);
    return Apply( chns, results, confidence);
}

void softcascade::setDebug( bool m_d )
{
    m_debug = m_d;
}
