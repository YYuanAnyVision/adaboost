#ifndef ADABOOST_HPP
#define ADABOOST_HPP
#include <iostream>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "../binaryTree/binarytree.hpp"

using namespace cv;
using namespace std;

class Adaboost
{
	public:

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
					Mat &predicted_vector);				/*out: predicted vector, double format, predicted confidence */



		/* 
		 * ===  FUNCTION  ======================================================================
		 *         Name:  ApplyLabel
		 *  Description: same as Apply, but output the int label 
		 * =====================================================================================
		 */
		bool ApplyLabel( const Mat &test_data,			/*  in: test data format-> featuredim x numberSample */ 
						 Mat &predicted_label);			/*out: predicted vector, int format, predicted label */

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
		bool saveModel( string filename );

		
		/* 
		 * ===  FUNCTION  ======================================================================
		 *         Name:  loadModel
		 *  Description:  load the model from xml file
		 * =====================================================================================
		 */
		bool loadModel( string filename );

	private:
		vector<binaryTree> m_trees;
		bool m_debug;
		int  m_feature_dim;
};
#endif

