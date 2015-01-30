#ifndef ADABOOST_HPP
#define ADABOOST_HPP
#include <iostream>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "../binaryTree/binarytree.hpp"
#include "../log/tlog.h"

using namespace cv;
using namespace std;

class Adaboost
{
	public:
        TLog *pLog;

		/* 
		 * ===  FUNCTION  ======================================================================
		 *         Name:  Train
		 *  Description:  ~ train ~
		 *          out:  no error -> true
		 * =====================================================================================
		 */
		bool Train(  const Mat &neg_data,				/* in : neg data  format-> featuredim x number0 */
					 const Mat &pos_data,				/* in : pos data  format-> featuredim x number0 */
					 const int &nWeaks,					/* in : how many weak classifiers( decision tree) are used  */
					 const tree_para &treepara);		/* in : parameter for the decision tree */


		/* 
		 * ===  FUNCTION  ======================================================================
		 *         Name:  Apply
		 *  Description:  apply the model to the Test Data, output double confidence
		 * =====================================================================================
		 */
		bool Apply( const Mat &test_data,				/* in: test data format-> featuredim x numberSample */
					Mat &predicted_vector) const;		/*out: predicted vector, double format, predicted confidence */



		/* 
		 * ===  FUNCTION  ======================================================================
		 *         Name:  ApplyLabel
		 *  Description: same as Apply, but output the int label 
		 * =====================================================================================
		 */
		bool ApplyLabel( const Mat &test_data,			/*  in: test data format-> featuredim x numberSample */ 
						 Mat &predicted_label) const;	/*out: predicted vector, int format, predicted label */

		/* 
		 * ===  FUNCTION  ======================================================================
		 *         Name:  SetDebug
		 *  Description:  wanna Debug information ?
		 * =====================================================================================
		 */
		void SetDebug( bool yesIwant);
		

		/* 
		 * ===  FUNCTION  ======================================================================
		 *         Name:  saveModel
		 *  Description:  save the adaboost model
		 * =====================================================================================
		 */
		bool saveModel( string filename ) const;

		
		/* 
		 * ===  FUNCTION  ======================================================================
		 *         Name:  loadModel
		 *  Description:  load the model from xml file
		 * =====================================================================================
		 */
		bool loadModel( string filename );

		
		/* 
		 * ===  FUNCTION  ======================================================================
		 *         Name:  applyAndGetError
		 *  Description:  out: fn : false negative
		 *				  out: fp : false potitive
		 * =====================================================================================
		 */
		void applyAndGetError( const Mat &neg_data,			/* in : neg data  format-> featuredim x number0 */
							   const Mat &pos_data,			/* in : neg data  format-> featuredim x number0 */
							   double &fn,					/* out: false negative */
							   double &fp) const;			/* out: false positive */


		/* 
		 * ===  FUNCTION  ======================================================================
		 *         Name:  getTreesDepth
		 *  Description:  get the number of nodes of each tree, if all of them have the identical nodes
		 *				  than return the value, otherwisze -1;
		 *			out:  depth of the trees
		 * =====================================================================================
		 */
		int getTreesNodes() const;


		/* 
		 * ===  FUNCTION  ======================================================================
		 *         Name:  getMaxNUmNodes
		 *  Description:  return the max number of nodes among all the trees
		 * =====================================================================================
		 */
		int getMaxNumNodes() const;


		/* 
		 * ===  FUNCTION  ======================================================================
		 *         Name:  getTrees
		 *  Description:  return the trees for softcascade to combine them
		 * =====================================================================================
		 */
		const vector<binaryTree>& getTrees();


		/* 
		 * ===  FUNCTION  ======================================================================
		 *         Name:  getNodes
		 *  Description:  return the number of nodes of every tree
		 * =====================================================================================
		 */
		const Mat& getNodes();

	private:
		vector<binaryTree> m_trees;
		bool m_debug;
		int  m_feature_dim;
		Mat  m_nodes;						/* number_of_trees x1 : number of nodes for each tree */
};
#endif

