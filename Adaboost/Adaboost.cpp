#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include <algorithm>
#include "opencv2/highgui/highgui.hpp"
#include "Adaboost.hpp"

using namespace std;
using namespace cv;

bool Adaboost::Train(	const Mat &neg_data,				/* in : neg data format-> featuredim x number0*/
						const Mat &pos_data,				/* in : pos data format-> featuredim x number1*/
						const int &nWeaks,					/* in : how many weak classifiers( decision tree) are used  */
						const tree_para &treepara)			/* in : parameter for the decision tree */
{
	if( neg_data.rows != pos_data.rows || neg_data.type() != pos_data.type())
	{
		cout<<"In function Adaboost:Train : neg_data and pos_data should be the same type, and having the same rows( column feature vector)"<<endl;
		return false;
	}
	if(  neg_data.empty() || pos_data.empty() )
	{
		cout<<"In function Adaboost:Train : data empty "<<endl;
		return false;
	}

	/*  using the specific format, carrying weight and other information  */
	data_pack train_pack;
	train_pack.neg_data = neg_data;
	train_pack.pos_data = pos_data;

	/*  infos about the data  */
	int number_neg_samples = neg_data.cols;
	int number_pos_samples = pos_data.cols;
	int feature_dim		   = neg_data.rows;

	/* save the featuredim */
	m_feature_dim = feature_dim;

	/*  scores about the sample */
	Mat H0 = Mat::zeros( number_neg_samples, 1, CV_64F);
	Mat H1 = Mat::zeros( number_pos_samples, 1, CV_64F);
	
	/*  infos about training , errs and losses */
	Mat losses = Mat::zeros( nWeaks, 1, CV_64F);
	Mat errs   = Mat::zeros( nWeaks, 1, CV_64F);

	m_trees.reserve( nWeaks);
	/*  train each weak classifier */
	for ( int c=0;c<nWeaks ; c++) 
	{
		if(m_debug)
		{
			cout<<"\n\n\n"<<endl;
			cout<<"############################################## round "<<c<<" ##########################################"<<endl;
		}
		binaryTree bt; bt.SetDebug(m_debug);
		
		double time_single_shot = getTickCount();
		if(! bt.Train( train_pack , treepara ) )
		{
			cout<<"in fuction Adaboost:Train, error training tree No "<<c<<endl;
			return false;
		}
		time_single_shot = (double)getTickCount() - time_single_shot;
		if(m_debug)
			cout<<"time single train on binaryTree is "<<time_single_shot/(double)getTickFrequency()<<endl;

		/*  apply the training data on the model */
		Mat h0, h1;										/*  predicted labels */
		bt.Apply( neg_data, h0);
		bt.Apply( pos_data, h1);

		double alpha = 1; 
		double error = bt.getTrainError();

		/*  alpha = 0.5*log( (1-weightederror)/weightederror ) */
		alpha = std::max( -5.0, std::min(  5.0, 0.5*std::log((1-error)/error)  ));
		if(m_debug)
			cout<<"alpha is "<<alpha<<" , error is "<<error<<endl;

		/*  ----------- stopping early ----------- */
		if( alpha <= 0)
		{
			cout<<"stopping early, cause alpha less than zero "<<endl;
			break;
		}

		/* ~~~ incorporate directly into the tree model ~~~  weights the output by a factor alpha */
		bt.scaleHs( alpha );

		if(m_debug)
		{
			cout<<"Tree No "<<c<<" Infos: "<<endl;
			bt.showTreeInfo();
		}

		/* update cumulative scores H and weights */
		H0 = H0 + alpha*h0;							/* output of the clf, since F_t(x) = F_t-1(x) + alpha_t*h_t(x) */
		H1 = H1 + alpha*h1;

		/*  divide by 2--> since we operate neg and pos samples separatly( sum of each weight is 1 potentially) */
		cv::exp( H0, train_pack.wts0 );       train_pack.wts0 = train_pack.wts0/(2*number_neg_samples);
		cv::exp( -1.0*H1, train_pack.wts1 );  train_pack.wts1 = train_pack.wts1/(2*number_pos_samples);
		
		double loss = (cv::sum( train_pack.wts0))[0] + (cv::sum(train_pack.wts1))[0];
		/*  stop training if loss too small .... no need to continue */
		if( loss < 1e-40)
		{
			cout<<"stopping early, loss = "<<loss<<", less than 1e-40"<<endl;
			break;
		}
		
		/* otherwise adding the new weak classifer to the end */
		m_trees.push_back( bt );

		/*  show the training error after adding the weak classifier */

		errs.at<double>(c,0) = bt.getTrainError();
		losses.at<double>(c,0) = loss;
		/* output  training informations , checks what's going on inside */
		int verbose = 16;
		if((c+1)%verbose == 0)
		{
			double fn,fp;
			applyAndGetError(neg_data, pos_data, fn, fp );
			cout<<"Training FP is "<<setprecision(6)<<setw(12)<<fp<<", FN is "<<
				setprecision(6)<<setw(12)<<fn<<". size(learner) "<<setw(5)<<m_trees.size();
			cout<<"\terr="<<setprecision(6)<<setw(10)<<errs.at<double>(c,0)<<"\t loss ="<<losses.at<double>(c,0)<<endl;
		}
	}

	/*  records the number of nodes of each tree */
	m_nodes = Mat( m_trees.size(), 1, CV_32S);			/*  number of node of the tree */
	for( int c=0;c<m_trees.size();c++)
	{
		const biTree *ptr = m_trees[c].getTree();
		m_nodes.at<int>(c,0) = (*ptr).fids.cols;
	}

	/* --------------------------  output debug information ------------------------------ */
	if(m_debug)
	{
		cout<<"Total number of weak classifiers is "<<m_trees.size()<<endl;
		/*  computing false positive */
		double fp = 0, fn = 0;
		for(int c=0;c<H0.rows;c++)
			fp += (H0.at<double>(c,0) > 0?1:0);
		fp /= H0.rows;
		
		for( int c=0;c<H1.rows;c++)
			fn += ( H1.at<double>(c,0) <0?1:0);
		fn /= H1.rows;

		cout<<"Results on the training data is "<<endl;
		cout<<"--> False Positive is "<<fp<<endl;
		cout<<"--> False Negative is "<<fp<<endl;
		cout<<"\n"<<endl;
		cout<<"====================================== Finished ========================================="<<endl; 
		int verbose = 16;
		for ( int c=0;c<errs.rows ;c++ ) 
		{
			if((c+1)%verbose == 0)			/* only show part of the informations */
				cout <<setprecision(8)<< "errs in weak learner "<<setw(4)<<c<<"\terr="<<errs.at<double>(c,0)<<"\t loss ="<<losses.at<double>(c,0)<<endl;
		}
	}

	/*  save the trained model  */
	if(!saveModel("lastTrain.xml"))
	{
		cout<<"the model is saved as lastTrain.xml by default"<<endl;
		cout<<"can not save the model .."<<endl;
		return false;
	}
	return true;
}


void Adaboost::SetDebug( bool d )
{
	m_debug = d;
}


bool Adaboost::Apply( const Mat &test_data,				/*  in: test data format-> featuredim x numberSample */
					Mat &predicted_vector) const		/* out: predicted vector, double format, predicted confidence */
{
	if( m_feature_dim != test_data.rows)
	{
		cout<<"input Dimension Wrong, or you should use column feature "<<endl;
		return false;
	}
	if( m_trees.empty())
	{
		cout<<"tree ne ? tree ne? tree doumeiyou ,nishuo ge mao"<<endl;
		return false;
	}
	
	predicted_vector = Mat::zeros( test_data.cols, 1, CV_64F);
	for( int c=0;c<m_trees.size();c++)
	{
		Mat p; m_trees[c].Apply( test_data, p);
		predicted_vector += p;
	}
	return true;
}

bool Adaboost::ApplyLabel( const Mat &test_data,			/*  in: test data format-> featuredim x numberSample */ 
				Mat &predicted_label)	const				/*out: predicted vector, int format, predicted label */
{
	Mat predicted_confidence;
	if( !Apply( test_data, predicted_confidence ))
	{
		cout<<"Wrong predict the label "<<endl;
		return false;
	}
	
	predicted_label = Mat( predicted_confidence.size(), CV_32S);
	for( int c=0;c<predicted_confidence.rows;c++)
	{
		if(predicted_confidence.at<double>(c,0) > 0 )
			predicted_label.at<int>(c,0) = 1;
		else
			predicted_label.at<int>(c,0) = -1;
	}
	return true;
}

int Adaboost::getTreesDepth() const
{
	int nn = m_nodes.at<int>(0,0);
	for ( int c=0;c<m_nodes.rows ;c++ ) 
	{
		if( m_nodes.at<int>(c,0) != nn)
			return -1;
	}
	return nn;
}

bool Adaboost::saveModel( string filename ) const
{
	cout<<"saving the model ..."<<endl;
	if(m_trees.empty())
	{
		cout<<"model is empty ..."<<endl;
		return false;
	}
	/* put all the infos into a big matrix , then save it
	 * each row -> one model's parameters*/

	/*  take a sample , measure the length */
	const biTree *sample = m_trees[0].getTree();
	int   number_of_trees = m_trees.size();

	/* allocate the mem, not every tree have the identical depth and nodes, some of them stop spliting early
	 *  results less nodes,  record the nodes and the model data*/
	int max_number_nodes = 0;
	for( int c=0;c<m_nodes.rows;c++)
	{
		max_number_nodes = std::max( max_number_nodes, m_nodes.at<int>(c,0) );
	}

	Mat fids_pack		= Mat::zeros(   number_of_trees, max_number_nodes,(*sample).fids.type() );
	Mat child_pack		= Mat::zeros(   number_of_trees, max_number_nodes,(*sample).child.type() );
	Mat thrs_pack		= Mat::zeros(   number_of_trees, max_number_nodes,(*sample).thrs.type() );
	Mat hs_pack			= Mat::zeros(   number_of_trees, max_number_nodes,(*sample).hs.type() );
	Mat weights_pack	= Mat::zeros(   number_of_trees, max_number_nodes,(*sample).weights.type() );
	Mat depth_pack		= Mat::zeros(   number_of_trees, max_number_nodes,(*sample).depth.type() );

	for( int c=0;c<m_trees.size();c++)
	{
		const biTree *ptr = m_trees[c].getTree();
		int nnodes = (*ptr).fids.cols;
		(*ptr).fids.copyTo( fids_pack.row(c).colRange(0,nnodes));		/* copying the rigth data to the right place */
		(*ptr).thrs.copyTo( thrs_pack.row(c).colRange(0,nnodes));
		(*ptr).child.copyTo( child_pack.row(c).colRange(0,nnodes));
		(*ptr).hs.copyTo( hs_pack.row(c).colRange(0,nnodes));
		(*ptr).weights.copyTo( weights_pack.row(c).colRange(0,nnodes));
		(*ptr).depth.copyTo(depth_pack.row(c).colRange(0,nnodes));
	}

	FileStorage fs( filename, FileStorage::WRITE);
	if( !fs.isOpened())
	{
		cout<<"can create the model xml file .."<<endl;
		return false;
	}

	/* writing data */
	fs<<"nodes"<<m_nodes;
	fs<<"fids"<<fids_pack;
	fs<<"child"<<child_pack;
	fs<<"hs"<<hs_pack;
	fs<<"weights"<<weights_pack;
	fs<<"depth"<<depth_pack;
	fs<<"thrs"<<thrs_pack;
	fs<<"featureDim"<<m_feature_dim;
	cout<<"saving done! "<<endl;
	return true;
}

bool Adaboost::loadModel( string filename )
{
	/*  clear the trees first */
	m_trees.clear();
	cout<<"Loading model file "<<endl;
	FileStorage fs( filename, FileStorage::READ );
	if( !fs.isOpened())
	{
		cout<<"can not open file "<<filename<<endl;
		return false;
	}

	Mat fids_pack;			/*    ------ tree model ------       */
	Mat child_pack;			/*	 number_of_trees x node number   */
	Mat thrs_pack;			/*       see binaryTree.hpp		     */
	Mat hs_pack;			/*								     */
	Mat weights_pack;		/*     						         */
	Mat depth_pack;			/*    ------ tree model ------       */
	
	fs["nodes"] >> m_nodes;
	fs["fids"] >> fids_pack;
	fs["child"] >> child_pack;
	fs["hs"] >> hs_pack;
	fs["weights"] >> weights_pack;
	fs["depth"] >> depth_pack;
	fs["thrs"] >> thrs_pack;
	fs["featureDim"] >> m_feature_dim;

	cout<<"Loading model file done, now initialize the Models "<<endl;
	
	/*  assign the parameter to each tree model  */
	int number_of_trees = m_nodes.rows;
	m_trees.reserve( number_of_trees );
	for ( int c=0;c<number_of_trees ;c++ )
	{
		biTree bt;
		/*  binaryTree assumes the memory of the tree model is continuous ( pointer operation ) 
		 *  so it have to be the row format~~*/
		bt.fids		= fids_pack.row(c).colRange(0,m_nodes.at<int>(c,0));				/*  no memory copy here ~! */
		bt.child	= child_pack.row(c).colRange(0,m_nodes.at<int>(c,0));
		bt.weights	= weights_pack.row(c).colRange(0,m_nodes.at<int>(c,0));
		bt.hs		= hs_pack.row(c).colRange(0,m_nodes.at<int>(c,0));		
		bt.thrs     = thrs_pack.row(c).colRange(0,m_nodes.at<int>(c,0));
		bt.depth	= depth_pack.row(c).colRange(0,m_nodes.at<int>(c,0));	

		binaryTree bbt;
		bbt.setTreeModel(bt);
		m_trees.push_back( bbt);
	}
	cout<<"m_trees's size is "<<m_trees.size()<<endl;
	cout<<"Loading Model done!"<<endl;
	return true;
}

void Adaboost::applyAndGetError( const Mat &neg_data,			/* in : neg data  format-> featuredim x number0 */
							     const Mat &pos_data,			/* in : neg data  format-> featuredim x number0 */
							     double &fn,					/* out: false negative */
							     double &fp) const				/* out: false positive */
{
	Mat predicted_label0, predicted_label1;
	ApplyLabel( neg_data, predicted_label0);
	ApplyLabel( pos_data, predicted_label1);
	
	fp = 0;
	fn = 0;
	for(int c=0;c<predicted_label0.rows;c++)
		fp += (predicted_label0.at<int>(c,0) > 0?1:0);
	fp /= predicted_label0.rows;

	for(int c=0;c<predicted_label1.rows;c++)
		fn += (predicted_label1.at<int>(c,0) < 0?1:0);
	fn /= predicted_label1.rows;

}
