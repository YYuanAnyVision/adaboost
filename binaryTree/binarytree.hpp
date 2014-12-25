/*-----------------------------------------------------------------------------
 *  Author:			yuanyang
 *  Date:			20141225
 *  Description:	binary tree
 *-----------------------------------------------------------------------------*/

#ifndef BINARYTREE_HPP
#define BINARYTREE_HPP
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

/*  parameters for one tree */
struct tree_para
{
	int		nBins;		/* maximum number of quantization bins, better <=256 */
	int		maxDepth;   /* max depth of the tree */
	double	minWeight;	/* minimum sample weight allow split*/
	double	fracFtrs;	/* fraction of features to sample for each node split */
	int		nThreads;	/* max number of computational threads to use */

	tree_para()
	{
		nBins = 256;
		maxDepth = 2;
		minWeight = 0.01;
		fracFtrs = 0.0625;
		nThreads = 8;
	}
}


/*  struct of the binary tree , save and load */
struct biTree
{

	Mat fids;		/* Kx1 feature index for each node , K for number of nodes*/
	Mat thrs;		/* Kx1 thresholds for each node */
	Mat child;		/* Kx1 child index for each node */
	Mat hs;			/* Kx1 log ratio (.5*log(p/(1-p)) at each node, used later to decide polarity */
	Mat weights;	/* Kx1 total sample weight at each node */
	Mat depth;		/* Kx1 depth of node*/
}


class binaryTree
{
	
	public:
		/* 
		 * ===  FUNCTION  ======================================================================
		 *         Name:  Train
		 *  Description:  train the binary tree
		 *			out:  true->no error
		 * =====================================================================================
		 */
		bool Train( 
				const Mat &neg_data,			/* input, format-> numbers0 x featuredim */
			    const Mat &pos_data,			/* input, format-> numbers1 x featuredim */ 
			    const tree_para &paras			/* input tree paras */
			   );
	private:

		/* 
		 * ===  FUNCTION  ======================================================================
		 *         Name:  checkTreeParas
		 *  Description:  check if the parameters is right
		 *			out:  true if the parameter is in the right range
		 * =====================================================================================
		 */
		bool checkTreeParas( const tree_para & p );		/* input parameter */

};
#endif
