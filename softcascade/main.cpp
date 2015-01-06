#include <iostream>
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "softcascade.hpp"
#include "../Adaboost/Adaboost.hpp"
#include "../misc/misc.hpp"

using namespace std;
using namespace cv;

/* detector parameter define */
int main( int argc, char** argv)
{

	/*-----------------------------------------------------------------------------
	 *  Step 1 : set and check parameters
	 *-----------------------------------------------------------------------------*/

	cascadeParameter cas_para;

	/*-----------------------------------------------------------------------------
	 *  Step 2 : iterate bootstraping and training
	 *-----------------------------------------------------------------------------*/
	for( int stage=0;stage<cas_para.nWeaks.size();stage++)
	{
		TickMeter tk;
		tk.start();
		cout<<"Training Stage No "<<stage<<endl;
		/* TODO sample positives and compute info about channels */
		//if( stages == 0)
		//{
		//}

		/* TODO compute local decorrelation filters if needed */

		/* TODO compute lambdas */

		/* TODO compute features for positives */

		/* TODO sample negatives and compute features */

		/* TODO accumulate negatives from previous stages */

		/* TODO train boosted classifiers */
		tk.stop();
		cout<<"Done Stage No "<<stage<<" , time "<<tk.getTimeSec()<<endl;
	}

}
