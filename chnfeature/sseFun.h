#ifndef SSEFUN_H
#define SSEFUN_h
#include <math.h>
#include "sse.hpp"

#define PI 3.14159265358979323846264338


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  gradHist
 *  Description:  compute nOrients gradient histograms per bin x bin block of pixels
 * =====================================================================================
 */
void gradHist( const float *Mag,	// in : magnitude, size width x height
		       const float *Ori,	// in : oritentation, same size as magnitude
			   float *gHist,		// out: gradHist,  size ( width/binSize x nOrients ) x height/binSize, big matrix
			   int height,			// in : height
			   int width,			// in : width
			   int binSize,			// in : size of spatial bin, degree of aggregation,  eg : 4, 
			   int nOrients,		// in : number of orientation, eg : 6
			   int softBin=0,		// in : only softBin == 0 is supported now
			   bool full=false);	// in : true -> 0-2pi, false -> 0-pi			



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  conTri1Y
 *  Description:  convolve one row of I by a [1 p 1] filter (uses SSE)
 * =====================================================================================
 */
void convTri1Y( const float *InputData,		// in : input data
				float *OutputData,			// out: output data
				int width,					// in : length of this row( width of the image )
				float p,					// in : 
				int s=1);					// in : resample factor, only 1 or 2 is supported


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  conTri1
 *  Description:   convolve I by a [1 p 1] filter (uses SSE)
 * =====================================================================================
 */
void convTri1( const float *InputData,		// in : input data
			   float *OutputData,			// out: output data
			   int height,					// in : the height of the image
			   int width,					// in : the width of the image
			   int dim,						// in : dim is 1 for single channel image, 3 for color image
			   float p,						// in :
			   int s=1);					// in : resample factor, only 1 or 2 is supported


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  conTriY
 *  Description:  convolve one row of I by a 2rx1 triangle filter
 * =====================================================================================
 */
void convTriY( float *I, 
			   float *O, 
			   int w, 
			   int r, 
			   int s) ;

void convTri_sse( const float *I, float *O, int width, int height, int r,int d = 1, int s=1 );


// compute x and y gradients for just one row (uses sse)
void grad1( const float *I,   //in :data
            float *Gx,  //out: gradient of x
            float *Gy,  //out: gradient of y
            int h,      //in : height
            int w,      //in : width
            int x );     //in : index of row

float* acosTable();

// compute gradient magnitude and orientation at each location (uses sse)
void gradMag( float *I, float *M, float *O, int h, int w, int d, bool full );


// normalize gradient magnitude at each location (uses sse)
void gradMagNorm( float *M,                     // output: M = M/(S + norm)
                  const float *S,               // input : Source Matrix
                  int h, int w, float norm );    // input : parameters


// helper for gradHist, quantize O and M into O0, O1 and M0, M1 (uses sse)
void gradQuantize( const float *O, const float *M, int *O0, int *O1, float *M0, float *M1,
                   int nb, int n, float norm, int nOrients, bool full, bool interpolate );




// Constants for rgb2luv conversion and lookup table for y-> l conversion
template<class oT> oT* rgb2luv_setup( oT z, oT *mr, oT *mg, oT *mb,
                                      oT &minu, oT &minv, oT &un, oT &vn )
{
    // set constants for conversion
    const oT y0=(oT) ((6.0/29)*(6.0/29)*(6.0/29));
    const oT a= (oT) ((29.0/3)*(29.0/3)*(29.0/3));
    un=(oT) 0.197833; vn=(oT) 0.468331;
    mr[0]=(oT) 0.430574*z; mr[1]=(oT) 0.222015*z; mr[2]=(oT) 0.020183*z;
    mg[0]=(oT) 0.341550*z; mg[1]=(oT) 0.706655*z; mg[2]=(oT) 0.129553*z;
    mb[0]=(oT) 0.178325*z; mb[1]=(oT) 0.071330*z; mb[2]=(oT) 0.939180*z;
    oT maxi=(oT) 1.0/270; minu=-88*maxi; minv=-134*maxi;
    // build (padded) lookup table for y->l conversion assuming y in [0,1]
    static oT lTable[1064]; static bool lInit=false;
    if( lInit ) return lTable; oT y, l;
    for(int i=0; i<1025; i++) {
        y = (oT) (i/1024.0);
        l = y>y0 ? 116*(oT)pow((double)y,1.0/3.0)-16 : y*a;
        lTable[i] = l*maxi;
    }
    for(int i=1025; i<1064; i++) lTable[i]=lTable[i-1];
    lInit = true; return lTable;
}


// Convert from rgb to luv
template<class iT, class oT> void rgb2luv( const iT *I, 
                                           oT *J,
                                           int n,
                                           oT nrm )
{
    oT minu, minv, un, vn, mr[3], mg[3], mb[3];
    oT *lTable = rgb2luv_setup(nrm,mr,mg,mb,minu,minv,un,vn);
    oT *L=J, *U=L+n, *V=U+n;
    const iT *R=I+2, *G=I+1, *B=I;			// opencv , B,G,R,B,G,R..
    for( int i=0; i<n; i++ )
    {
        oT r, g, b, x, y, z, l;
        r=(oT)*R; R=R+3;
        g=(oT)*G; G=G+3;
        b=(oT)*B; B=B+3;
        x = mr[0]*r + mg[0]*g + mb[0]*b;
        y = mr[1]*r + mg[1]*g + mb[1]*b;
        z = mr[2]*r + mg[2]*g + mb[2]*b;
        l = lTable[(int)(y*1024)];
        *(L++) = l; z = 1/(x + 15*y + 3*z + (oT)1e-35);
        *(U++) = l * (13*4*x*z - 13*un) - minu;
        *(V++) = l * (13*9*y*z - 13*vn) - minv;
    }
}

// Convert from rgb to luv using sse
template<class iT> void rgb2luv_sse( iT *I, float *J, int n, float nrm ) 
{
    const int k=256; float R[k], G[k], B[k];
    if( (size_t(R)&15||size_t(G)&15||size_t(B)&15||size_t(I)&15||size_t(J)&15)
            || n%4>0 )
    {
        rgb2luv(I,J,n,nrm); return;
    }                      // data not align
    int i=0, i1, n1; float minu, minv, un, vn, mr[3], mg[3], mb[3];
    float *lTable = rgb2luv_setup(nrm,mr,mg,mb,minu,minv,un,vn);
    while( i<n )
    {
        n1 = i+k; if(n1>n) n1=n; float *J1=J+i; float *R1, *G1, *B1;
        /* ------------ RGB is now RRRRRGGGGGBBBBB ----------*/
        // convert to floats (and load input into cache)
        R1=R; G1=G; B1=B;
        iT *Bi=I+i*3, *Gi=Bi+1, *Ri=Bi+2;
        for( i1=0; i1<(n1-i); i1++ )
        {
            R1[i1] = (float) (*Ri);Ri = Ri+3;
            G1[i1] = (float) (*Gi);Gi = Gi+3;
            B1[i1] = (float) (*Bi);Bi = Bi+3;
        }
        /* ------------ RGB is now RRRRRGGGGGBBBBB ----------*/
        // compute RGB -> XYZ
        for( int j=0; j<3; j++ )
        {
            __m128 _mr, _mg, _mb, *_J=(__m128*) (J1+j*n);
            __m128 *_R=(__m128*) R1, *_G=(__m128*) G1, *_B=(__m128*) B1;
            _mr=SET(mr[j]); _mg=SET(mg[j]); _mb=SET(mb[j]);
            for( i1=i; i1<n1; i1+=4 )
            {
                *(_J++) = ADD( ADD(MUL(*(_R++),_mr),MUL(*(_G++),_mg)),MUL(*(_B++),_mb));
            }
        }
        /* ---------------XXXXXXXYYYYYYYZZZZZZZZ now --------------- */

        { // compute XZY -> LUV (without doing L lookup/normalization)
            __m128 _c15, _c3, _cEps, _c52, _c117, _c1024, _cun, _cvn;
            _c15=SET(15.0f); _c3=SET(3.0f); _cEps=SET(1e-35f);
            _c52=SET(52.0f); _c117=SET(117.0f), _c1024=SET(1024.0f);
            _cun=SET(13*un); _cvn=SET(13*vn);
            __m128 *_X, *_Y, *_Z, _x, _y, _z;
            _X=(__m128*) J1; _Y=(__m128*) (J1+n); _Z=(__m128*) (J1+2*n);
            for( i1=i; i1<n1; i1+=4 )
            {
                _x = *_X; _y=*_Y; _z=*_Z;
                _z = RCP(ADD(_x,ADD(_cEps,ADD(MUL(_c15,_y),MUL(_c3,_z)))));
                *(_X++) = MUL(_c1024,_y);
                *(_Y++) = SUB(MUL(MUL(_c52,_x),_z),_cun);
                *(_Z++) = SUB(MUL(MUL(_c117,_y),_z),_cvn);
            }
        }
        { // perform lookup for L and finalize computation of U and V
            for( i1=i; i1<n1; i1++ ) J[i1] = lTable[(int)J[i1]];
            __m128 *_L, *_U, *_V, _l, _cminu, _cminv;
            _L=(__m128*) J1; _U=(__m128*) (J1+n); _V=(__m128*) (J1+2*n);
            _cminu=SET(minu); _cminv=SET(minv);
            for( i1=i; i1<n1; i1+=4 ) {
                _l = *(_L++);
                *_U = SUB(MUL(_l,*_U),_cminu); _U++;
                *_V = SUB(MUL(_l,*_V),_cminv); _V++;
            }
        }
        i = n1;
    }
}


#endif
