#ifndef SOFTCASCADE
#define SOFTCASCADE

#include <string>
#include <iostream>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "../Adaboost/Adaboost.hpp"
#include "../binaryTree/binarytree.hpp"

using namespace std;
using namespace cv;


struct cascadeParameter
{
	vector<int> filter;					/* de-correlation filter parameters, eg [5 5]  */
	Size modelDs;						/* model height+width without padding (eg [100 41]) */
	Size modelDsPad;					/* model height+width with padding (eg [128 64])*/

	/* ---> TODO non maximun suppression parameters ... */

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

	/* ---> TODO jitter parameters .... */
	cascadeParameter()
	{
		modelDs    = Size(41, 100);
		modelDsPad = Size(64, 128);
		stride     = 4;
		cascThr	   = -1;
		cascCal    = 0.005;
		nWeaks.push_back( 128);
		pBoost_nweaks = 128;
		pBoost_pTree = tree_para();
		infos = "no infos";
		nPos = -1;
		nNeg = 5000;
		nPerNeg = 25;
		nAccNeg = 10000;
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
		 *          out:  predicted_result
		 * =====================================================================================
		 */
		bool Apply( const Mat &input_data,		/*  in: featuredim x number_of_samples */
				    Mat &predicted_result );	/* out: number_of_samples x 1 */

		
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


		private:
			Mat m_fids;			/* nxK feature index for each node , n -> number of trees, K -> number of nodes*/
			Mat m_thrs;			/* nxK thresholds for each node */
			Mat m_child;		/* nxK child index for each node */
			Mat m_hs;			/* nxK log ratio (.5*log(p/(1-p)) at each node  */
			Mat m_weights;		/* nxK total sample weight at each node */
			Mat m_depth;		/* nxK depth of node*/

};
#endif
