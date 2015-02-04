#include <math.h>
#include <string.h>

#include "sseFun.h"
#include "wrappers.hpp"
#define PI 3.14159265358979323846264338

// compute nOrients gradient histograms per bin x bin block of pixels
void gradHist( const float *M,const float *O, float *H, int h, int w, int bin, int nOrients, int softBin, bool full )
{
    const int hb=h/bin, wb=w/bin, h0=hb*bin, w0=wb*bin, nb=wb*hb;
    const float s=(float)bin, sInv=1/s, sInv2=1/s/s;
    float *H0, *H1, *M0, *M1; int x, y; int *O0, *O1; float xb, init;

    O0=(int*)alMalloc(w*sizeof(int),16); M0=(float*) alMalloc(w*sizeof(float),16);
    O1=(int*)alMalloc(w*sizeof(int),16); M1=(float*) alMalloc(w*sizeof(float),16);

    // main loop
    for( x=0; x<h0; x++ )
    {
        // compute target orientation bins for entire column - very fast
        gradQuantize(O+x*w,M+x*w,O0,O1,M0,M1,nb,w0,sInv2,nOrients,full,softBin>=0);
        if( softBin<0 && softBin%2==0 ) {
            // no interpolation w.r.t. either orienation or spatial bin
            H1=H+(x/bin)*wb;
#define GH H1[O0[y]]+=M0[y]; y++;
            if( bin==1 )      for(y=0; y<w0;) { GH; H1++; }
            else if( bin==2 ) for(y=0; y<w0;) { GH; GH; H1++; }
            else if( bin==3 ) for(y=0; y<w0;) { GH; GH; GH; H1++; }
            else if( bin==4 ) for(y=0; y<w0;) { GH; GH; GH; GH; H1++; }
            else for( y=0; y<w0;) { for( int y1=0; y1<bin; y1++ ) { GH; } H1++; }
#undef GH

        } else if( softBin%2==0 || bin==1 ) {
            // interpolate w.r.t. orientation only, not spatial bin
            H1=H+(x/bin)*wb;
#define GH H1[O0[y]]+=M0[y]; H1[O1[y]]+=M1[y]; y++;
            if( bin==1 )      for(y=0; y<w0;) { GH; H1++; }
            else if( bin==2 ) for(y=0; y<w0;) { GH; GH; H1++; }
            else if( bin==3 ) for(y=0; y<w0;) { GH; GH; GH; H1++; }
            else if( bin==4 ) for(y=0; y<w0;) { GH; GH; GH; GH; H1++; }
            else for( y=0; y<w0;) { for( int y1=0; y1<bin; y1++ ) { GH; } H1++; }
#undef GH
        }
    }
    alFree(O0); alFree(O1); alFree(M0); alFree(M1);
    // normalize boundary bins which only get 7/8 of weight of interior bins
    if( softBin%2!=0 ) for( int o=0; o<nOrients; o++ ) {
        x=0; for( y=0; y<hb; y++ ) H[o*nb+x+y*wb]*=8.f/7.f;
        y=0; for( x=0; x<wb; x++ ) H[o*nb+x+y*wb]*=8.f/7.f;
        x=wb-1; for( y=0; y<hb; y++ ) H[o*nb+x+y*wb]*=8.f/7.f;
        y=hb-1; for( x=0; x<wb; x++ ) H[o*nb+x+y*wb]*=8.f/7.f;
    }
}


// helper for gradHist, quantize O and M into O0, O1 and M0, M1 (uses sse)
void gradQuantize( const float *O, const float *M, int *O0, int *O1, float *M0, float *M1,
                   int nb, int n, float norm, int nOrients, bool full, bool interpolate )
{
    // assumes all *OUTPUT* matrices are 4-byte aligned
    int i, o0, o1; float o, od, m;
    __m128i _o0, _o1, *_O0, *_O1; __m128 _o, _od, _m, *_M0, *_M1;
    // define useful constants
    const float oMult=(float)nOrients/(full?2*(float)PI:(float)PI); const int oMax=nOrients*nb;
    const __m128 _norm=SET(norm), _oMult=SET(oMult), _nbf=SET((float)nb);
    const __m128i _oMax=SET(oMax), _nb=SET(nb);
    // perform the majority of the work with sse
    _O0=(__m128i*) O0; _O1=(__m128i*) O1; _M0=(__m128*) M0; _M1=(__m128*) M1;
    if( interpolate ) for( i=0; i<=n-4; i+=4 ) {
        _o=MUL(LDu(O[i]),_oMult); _o0=CVT(_o); _od=SUB(_o,CVT(_o0));
        _o0=CVT(MUL(CVT(_o0),_nbf)); _o0=AND(CMPGT(_oMax,_o0),_o0); *_O0++=_o0;
        _o1=ADD(_o0,_nb); _o1=AND(CMPGT(_oMax,_o1),_o1); *_O1++=_o1;
        _m=MUL(LDu(M[i]),_norm); *_M1=MUL(_od,_m); *_M0++=SUB(_m,*_M1); _M1++;
    } else for( i=0; i<=n-4; i+=4 ) {
        _o=MUL(LDu(O[i]),_oMult); _o0=CVT(ADD(_o,SET(.5f)));
        _o0=CVT(MUL(CVT(_o0),_nbf)); _o0=AND(CMPGT(_oMax,_o0),_o0); *_O0++=_o0;
        *_M0++=MUL(LDu(M[i]),_norm); *_M1++=SET(0.f); *_O1++=SET(0);
    }
    // compute trailing locations without sse
    if( interpolate ) for(; i<n; i++ ) {
        o=O[i]*oMult; o0=(int) o; od=o-o0;
        o0*=nb; if(o0>=oMax) o0=0; O0[i]=o0;
        o1=o0+nb; if(o1==oMax) o1=0; O1[i]=o1;
        m=M[i]*norm; M1[i]=od*m; M0[i]=m-M1[i];
    } else for(; i<n; i++ ) {
        o=O[i]*oMult; o0=(int) (o+.5f);
        o0*=nb; if(o0>=oMax) o0=0; O0[i]=o0;
        M0[i]=M[i]*norm; M1[i]=0; O1[i]=0;
    }
}



// convolve one row of I by a [1 p 1] filter (uses SSE)
void convTri1Y( const float *I, float *O, int w, float p, int s ) {
#define C4(m,o) ADD(ADD(LDu(I[m*j-1+o]),MUL(p,LDu(I[m*j+o]))),LDu(I[m*j+1+o]))
    int j=0, k=((~((size_t) O) + 1) & 15)/4, h2=(w-1)/2;
    if( s==2 )
    {
        for( ; j<k; j++ ) O[j]=I[2*j]+p*I[2*j+1]+I[2*j+2];
        for( ; j<h2-4; j+=4 ) STR(O[j],_mm_shuffle_ps(C4(2,1),C4(2,5),136));
        for( ; j<h2; j++ ) O[j]=I[2*j]+p*I[2*j+1]+I[2*j+2];
        if( w%2==0 ) O[j]=I[2*j]+(1+p)*I[2*j+1];
    }
    else
    {
        O[j]=(1+p)*I[j]+I[j+1]; j++; if(k==0) k=(w<=4) ? w-1 : 4;
        for( ; j<k; j++ ) O[j]=I[j-1]+p*I[j]+I[j+1];
        for( ; j<w-4; j+=4 ) STR(O[j],C4(1,0));
        for( ; j<w-1; j++ ) O[j]=I[j-1]+p*I[j]+I[j+1];
        O[j]=I[j-1]+(1+p)*I[j];
    }
#undef C4
}

// convolve I by a [1 p 1] filter (uses SSE)
void convTri1( const float *I, float *O, int h, int w, int d, float p, int s) {
    const float nrm = 1.0f/((p+2)*(p+2)); int i, j, w0=w-(w%4);
    const float *Il, *Im, *Ir;
    float *T=(float*) alMalloc(w*sizeof(float),16);
    for( int d0=0; d0<d; d0++ )
        for( i=s/2; i<h; i+=s )
        {
            Il=Im=Ir=I+i*w+d0*h*w; if(i>0) Il-=w; if(i<h-1) Ir+=w;
            for( j=0; j<w0; j+=4 )
                STR(T[j],MUL(nrm,ADD(ADD(LDu(Il[j]),MUL(p,LDu(Im[j]))),LDu(Ir[j]))));
            for( j=w0; j<w; j++ ) T[j]=nrm*(Il[j]+p*Im[j]+Ir[j]);
            convTri1Y(T,O,w,p,s); O+=w/s;
        }
    alFree(T);
}


// compute x and y gradients for just one row (uses sse)
void grad1( const float *I,   //in :data
            float *Gx,  //out: gradient of x
            float *Gy,  //out: gradient of y
            int h,      //in : height
            int w,      //in : width
            int x )     //in : index of row
{
    int y, y1;
    const float *Ip, *In; float r; __m128 *_Ip, *_In, *_G, _r;
    //compute row of Gy
    Ip=I-w; In=I+w; r=.5f;
    if(x==0) { r=1; Ip+=w; } else if(x==h-1) { r=1; In-=w; }      //on the border
    if( w<4 || w%4>0 || (size_t(I)&15) || (size_t(Gy)&15) ) {     //data align?
        for( int c=0; c<w; c++ ) *Gy++=(*In++-*Ip++)*r;
    } else {
        _G=(__m128*) Gy; _Ip=(__m128*) Ip; _In=(__m128*) In; _r = SET(r);
        for(int c=0; c<w; c+=4) *_G++=MUL(SUB(*_In++,*_Ip++),_r);
    }

    // compute row of Gx
#define GRADX(r) *Gx++=(*In++-*Ip++)*r;
    Ip=I; In=Ip+1;
    // equivalent --> GRADX(1); Ip--; for(y=1; y<w-1; y++) GRADX(.5f); In--; GRADX(1);
    y1=((~((size_t) Gx) + 1) & 15)/4; if(y1==0) y1=4; if(y1>w-1) y1=w-1;      // y1 -> the number of element with out using sse
    GRADX(1); Ip--; for(y=1; y<y1; y++) GRADX(.5f);
    _r = SET(.5f); _G=(__m128*) Gx;
    for(; y+4<w-1; y+=4, Ip+=4, In+=4, Gx+=4)
        *_G++=MUL(SUB(LDu(*In),LDu(*Ip)),_r);
    for(; y<w-1; y++) GRADX(.5f); In--; GRADX(1);
#undef GRADX
}

float* acosTable() {
    const int n=10000, b=10; int i;
    static float a[n*2+b*2]; static bool init=false;
    float *a1=a+n+b; if( init ) return a1;
    for( i=-n-b; i<-n; i++ )   a1[i]=(float)PI;
    for( i=-n; i<n; i++ )      a1[i]=float(acos(i/float(n)));
    for( i=n; i<n+b; i++ )     a1[i]=0;
    for( i=-n-b; i<n/10; i++ ) if( a1[i] > (float)PI-1e-6f ) a1[i]=(float)PI-1e-6f;
    init=true; return a1;
}

// compute gradient magnitude and orientation at each location (uses sse)
void gradMag( const float *I, float *M, float *O, int h, int w, int d, bool full )
{
    int x, y, y1, c, w4, s; float *Gx, *Gy, *M2; __m128 *_Gx, *_Gy, *_M2, _m;
    float *acost = acosTable(), acMult=10000.0f;

    // allocate memory for storing one row of output (padded so w4%4==0)
    w4=(w%4==0) ? w : w-(w%4)+4; s=d*w4*sizeof(float);
    M2=(float*) alMalloc(s,16); _M2=(__m128*) M2;
    Gx=(float*) alMalloc(s,16); _Gx=(__m128*) Gx;
    Gy=(float*) alMalloc(s,16); _Gy=(__m128*) Gy;

    // compute gradient magnitude and orientation for each column
    for( x=0; x<h; x++ )
    {
        // compute gradients (Gx, Gy) with maximum squared magnitude (M2)
        for(c=0; c<d; c++)          // compute for each channel, take the max value
        {
            grad1( I+x*w+c*w*h, Gx+c*w4, Gy+c*w4, h, w, x );
            for( y=0; y<w4/4; y++ )
            {
                y1=w4/4*c+y;
                _M2[y1]=ADD(MUL(_Gx[y1],_Gx[y1]),MUL(_Gy[y1],_Gy[y1]));
                if( c==0 ) continue; _m = CMPGT( _M2[y1], _M2[y] );
                _M2[y] = OR( AND(_m,_M2[y1]), ANDNOT(_m,_M2[y]) );
                _Gx[y] = OR( AND(_m,_Gx[y1]), ANDNOT(_m,_Gx[y]) );
                _Gy[y] = OR( AND(_m,_Gy[y1]), ANDNOT(_m,_Gy[y]) );
            }
        }
        // compute gradient mangitude (M) and normalize Gx // avoid the exception when arctan(Gy/Gx)
        for( y=0; y<w4/4; y++ ) {
            _m = SSEMIN( RCPSQRT(_M2[y]), SET(1e10f) );
            _M2[y] = RCP(_m);
            if(O) _Gx[y] = MUL( MUL(_Gx[y],_m), SET(acMult) );
            if(O) _Gx[y] = XOR( _Gx[y], AND(_Gy[y], SET(-0.f)) );
        };
        memcpy( M+x*w, M2, w*sizeof(float) );
        // compute and store gradient orientation (O) via table lookup
        if( O!=0 ) for( y=0; y<w; y++ ) O[x*w+y] = acost[(int)Gx[y]];
        if( O!=0 && full ) {
            y1=((~size_t(O+x*w)+1)&15)/4; y=0;
            for( ; y<y1; y++ ) O[y+x*w]+=(Gy[y]<0)*(float)PI;
            for( ; y<w-4; y+=4 ) STRu( O[y+x*w],
                    ADD( LDu(O[y+x*w]), AND(CMPLT(LDu(Gy[y]),SET(0.f)),SET((float)PI)) ) );
            for( ; y<w; y++ ) O[y+x*w]+=(Gy[y]<0)*(float)PI;
        }
    }

    alFree(Gx); alFree(Gy); alFree(M2);
}


// normalize gradient magnitude at each location (uses sse)
void gradMagNorm( float *M,                     // output: M = M/(S + norm)
                  const float *S,               // input : Source Matrix
                  int h, int w, float norm )    // input : parameters
{
    __m128 *_M, *_S, _norm; int i=0, n=h*w, n4=n/4;
    _S = (__m128*) S; _M = (__m128*) M; _norm = SET(norm);
    bool sse = !(size_t(M)&15) && !(size_t(S)&15);
    if(sse)
        for(; i<n4; i++)
        { *_M=MUL(*_M,RCP(ADD(*_S++,_norm))); _M++; }
    if(sse)
        i*=4;
    for(; i<n; i++) M[i] /= (S[i] + norm);
}

// convolve one row of I by a 2rx1 triangle filter
void convTriY( float *I, float *O, int w, int r, int s ) 
{
    r++; float t, u; int j, r0=r-1, r1=r+1, r2=2*w-r, h0=r+1, h1=w-r+1, h2=w;
    u=t=I[0]; for( j=1; j<r; j++ ) u+=t+=I[j]; u=2*u-t; t=0;
    if( s==1 ) {
        O[0]=u; j=1;
        for(; j<h0; j++) O[j] = u += t += I[r-j]  + I[r0+j] - 2*I[j-1];
        for(; j<h1; j++) O[j] = u += t += I[j-r1] + I[r0+j] - 2*I[j-1];
        for(; j<h2; j++) O[j] = u += t += I[j-r1] + I[r2-j] - 2*I[j-1];
    } else {
        int k=(s-1)/2; h2=(w/s)*s; if(h0>h2) h0=h2; if(h1>h2) h1=h2;
        if(++k==s) { k=0; *O++=u; } j=1;
        for(;j<h0;j++) { u+=t+=I[r-j] +I[r0+j]-2*I[j-1]; if(++k==s){ k=0; *O++=u; }}
        for(;j<h1;j++) { u+=t+=I[j-r1]+I[r0+j]-2*I[j-1]; if(++k==s){ k=0; *O++=u; }}
        for(;j<h2;j++) { u+=t+=I[j-r1]+I[r2-j]-2*I[j-1]; if(++k==s){ k=0; *O++=u; }}
    }
}

void convTri_sse( const float *I, float *O, int width, int height, int r,int d , int s) 
{
    r++; float nrm = 1.0f/(r*r*r*r); int i, j, k=(s-1)/2, h0, h1, w0;
    if(width%4==0)
        h0=h1=width;
    else
    { h0=width-(width%4); h1=h0+4; }
    w0=(height/s)*s;

    float *T=(float*) alMalloc(2*h1*sizeof(float),16), *U=T+h1;
    while( d-->0)
    {
        // initialize T and U
        for(j=0; j<h0; j+=4) STR(U[j], STR(T[j], LDu(I[j])));
        for(i=1; i<r; i++) for(j=0; j<h0; j+=4) INC(U[j],INC(T[j],LDu(I[j+i*width])));
        for(j=0; j<h0; j+=4) STR(U[j],MUL(nrm,(SUB(MUL(2,LD(U[j])),LD(T[j])))));
        for(j=0; j<h0; j+=4) STR(T[j],0);
        for(j=h0; j<width; j++ ) U[j]=T[j]=I[j];
        for(i=1; i<r; i++) for(j=h0; j<width; j++ ) U[j]+=T[j]+=I[j+i*width];
        for(j=h0; j<width; j++ ) { U[j] = nrm * (2*U[j]-T[j]); T[j]=0; }
        // prepare and convolve each column in turn
        k++; if(k==s) { k=0; convTriY(U,O,width,r-1,s); O+=width/s; }
        for( i=1; i<w0; i++ )
        {
            const float *Il=I+(i-1-r)*width; if(i<=r) Il=I+(r-i)*width; const float *Im=I+(i-1)*width;
            const float *Ir=I+(i-1+r)*width; if(i>height-r) Ir=I+(2*height-r-i)*width;
            for( j=0; j<h0; j+=4 ) {
                INC(T[j],ADD(LDu(Il[j]),LDu(Ir[j]),MUL(-2,LDu(Im[j]))));
                INC(U[j],MUL(nrm,LD(T[j])));
            }
            for( j=h0; j<width; j++ ) U[j]+=nrm*(T[j]+=Il[j]+Ir[j]-2*Im[j]);
            k++; if(k==s) { k=0; convTriY(U,O,width,r-1,s); O+=width/s; }
        }
        I+=width*height;
    }
    alFree(T);
}







