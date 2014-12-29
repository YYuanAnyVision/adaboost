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


		bool Apply();
		void SetDebug();
	private:
		vector<binaryTree> m_trees;
		bool m_debug;
};
#endif

