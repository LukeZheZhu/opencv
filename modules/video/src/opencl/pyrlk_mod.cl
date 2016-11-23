//__gen_ocl_force_simd8()

#define XLID get_local_id(0)
#define YLID get_local_id(1)
#define GID get_group_id(0)

#define SUBLID get_sub_group_local_id()

#define GRIDSIZE 3
#define LSx 3
#define LSy 6
#define BUFFER  (LSx*LSy)
#define BUFFER2 BUFFER>>1


// defeine local memory sizes
#define LM_W (LSx*GRIDSIZE+2)
#define LM_H (LSy*GRIDSIZE+2)

#ifndef WAVE_SIZE
#define WAVE_SIZE 1
#endif

#ifdef CPU

inline void reduce3(float val1, float val2, float val3,  __local float* smem1,  __local float* smem2,  __local float* smem3, int tid)
{
    smem1[tid] = val1;
    smem2[tid] = val2;
    smem3[tid] = val3;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = BUFFER2; i > 0; i >>= 1)
    {
        if(tid < i)
        {
            smem1[tid] += smem1[tid + i];
            smem2[tid] += smem2[tid + i];
            smem3[tid] += smem3[tid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

inline void reduce2(float val1, float val2, volatile __local float* smem1, volatile __local float* smem2, int tid)
{
    smem1[tid] = val1;
    smem2[tid] = val2;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = BUFFER2; i > 0; i >>= 1)
    {
        if(tid < i)
        {
            smem1[tid] += smem1[tid + i];
            smem2[tid] += smem2[tid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

inline void reduce1(float val1, volatile __local float* smem1, int tid)
{
    smem1[tid] = val1;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = BUFFER2; i > 0; i >>= 1)
    {
        if(tid < i)
        {
            smem1[tid] += smem1[tid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
#else
inline void reduce3(float val1, float val2, float val3,
             __local volatile float* smem1, __local volatile float* smem2, __local volatile float* smem3, int tid)
{
    smem1[tid] = val1;
    smem2[tid] = val2;
    smem3[tid] = val3;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 32)
    {
        smem1[tid] += smem1[tid + 32];
        smem2[tid] += smem2[tid + 32];
        smem3[tid] += smem3[tid + 32];
#if WAVE_SIZE < 32
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 16)
    {
#endif
        smem1[tid] += smem1[tid + 16];
        smem2[tid] += smem2[tid + 16];
        smem3[tid] += smem3[tid + 16];
#if WAVE_SIZE <16
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid<1)
    {
#endif
        local float8* m1 = (local float8*)smem1;
        local float8* m2 = (local float8*)smem2;
        local float8* m3 = (local float8*)smem3;
        float8 t1 = m1[0]+m1[1];
        float8 t2 = m2[0]+m2[1];
        float8 t3 = m3[0]+m3[1];
        float4 t14 = t1.lo + t1.hi;
        float4 t24 = t2.lo + t2.hi;
        float4 t34 = t3.lo + t3.hi;
        smem1[0] = t14.x+t14.y+t14.z+t14.w;
        smem2[0] = t24.x+t24.y+t24.z+t24.w;
        smem3[0] = t34.x+t34.y+t34.z+t34.w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

inline void reduce2(float val1, float val2, __local volatile float* smem1, __local volatile float* smem2, int tid)
{
    smem1[tid] = val1;
    smem2[tid] = val2;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 32)
    {
        smem1[tid] += smem1[tid + 32];
        smem2[tid] += smem2[tid + 32];
#if WAVE_SIZE < 32
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 16)
    {
#endif
        smem1[tid] += smem1[tid + 16];
        smem2[tid] += smem2[tid + 16];
#if WAVE_SIZE <16
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid<1)
    {
#endif
        local float8* m1 = (local float8*)smem1;
        local float8* m2 = (local float8*)smem2;
        float8 t1 = m1[0]+m1[1];
        float8 t2 = m2[0]+m2[1];
        float4 t14 = t1.lo + t1.hi;
        float4 t24 = t2.lo + t2.hi;
        smem1[0] = t14.x+t14.y+t14.z+t14.w;
        smem2[0] = t24.x+t24.y+t24.z+t24.w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

inline void reduce1(float val1, __local volatile float* smem1, int tid)
{
    smem1[tid] = val1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 32)
    {
        smem1[tid] += smem1[tid + 32];
#if WAVE_SIZE < 32
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 16)
    {
#endif
        smem1[tid] += smem1[tid + 16];
#if WAVE_SIZE <16
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid<1)
    {
#endif
        local float8* m1 = (local float8*)smem1;
        float8 t1 = m1[0]+m1[1];
        float4 t14 = t1.lo + t1.hi;
        smem1[0] = t14.x+t14.y+t14.z+t14.w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
#endif

#define SCALE (1.0f / (1 << 20))
#define	THRESHOLD	0.01f

// Image read mode
__constant sampler_t sampler    = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

// macro to get pixel value from local memory

#define VAL(_y,_x,_yy,_xx)    (IPatchLocal[mad24(((_y) + (_yy)), LM_W, ((_x) + (_xx)))])
inline void SetPatch(local float* IPatchLocal, int TileY, int TileX,
              float* Pch, float* Dx, float* Dy,
              float* A11, float* A12, float* A22, float w)
{
    int xid=get_local_id(0);
    int yid=get_local_id(1);
    int xBase = mad24(TileX, LSx, (xid + 1));
    int yBase = mad24(TileY, LSy, (yid + 1));

    *Pch = VAL(yBase,xBase,0,0);

    *Dx = mad((VAL(yBase,xBase,-1,1) + VAL(yBase,xBase,+1,1) - VAL(yBase,xBase,-1,-1) - VAL(yBase,xBase,+1,-1)), 3.0f, (VAL(yBase,xBase,0,1) - VAL(yBase,xBase,0,-1)) * 10.0f) * w;
    *Dy = mad((VAL(yBase,xBase,1,-1) + VAL(yBase,xBase,1,+1) - VAL(yBase,xBase,-1,-1) - VAL(yBase,xBase,-1,+1)), 3.0f, (VAL(yBase,xBase,1,0) - VAL(yBase,xBase,-1,0)) * 10.0f) * w;

    *A11 = mad(*Dx, *Dx, *A11);
    *A12 = mad(*Dx, *Dy, *A12);
    *A22 = mad(*Dy, *Dy, *A22);
}
#undef VAL

inline void GetPatch(image2d_t J, float x, float y,
              float* Pch, float* Dx, float* Dy,
              float* b1, float* b2)
{
    float diff = read_imagef(J, sampler, (float2)(x,y)).x-*Pch;
    *b1 = mad(diff, *Dx, *b1);
    *b2 = mad(diff, *Dy, *b2);
}

inline void GetError(image2d_t J, const float x, const float y, const float* Pch, float* errval)
{
    float diff = read_imagef(J, sampler, (float2)(x,y)).x-*Pch;
    *errval += fabs(diff);
}

#if 0
//macro to read pixel value into local memory.
#define READI(_y,_x) IPatchLocal[mad24(mad24((_y), LSy, yid), LM_W, mad24((_x), LSx, xid))] = read_imagef(I, sampler, (float2)(mad((float)(_x), (float)LSx, Point.x + xid - 0.5f), mad((float)(_y), (float)LSy, Point.y + yid - 0.5f))).x;
void ReadPatchIToLocalMem(image2d_t I, float2 Point, local float* IPatchLocal)
{
    int xid=get_local_id(0);
    int yid=get_local_id(1);
    //read (3*LSx)*(3*LSy) window. each macro call read LSx*LSy pixels block
    READI(0,0);READI(0,1);READI(0,2);
    READI(1,0);READI(1,1);READI(1,2);
    READI(2,0);READI(2,1);READI(2,2);
    if(xid<2)
    {// read last 2 columns border. each macro call reads 2*LSy pixels block
        READI(0,3);
        READI(1,3);
        READI(2,3);
    }

    if(yid<2)
    {// read last 2 row. each macro call reads LSx*2 pixels block
        READI(3,0);READI(3,1);READI(3,2);
    }

    if(yid<2 && xid<2)
    {// read right bottom 2x2 corner. one macro call reads 2*2 pixels block
        READI(3,3);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
#undef READI

#else

inline void ReduceSGM(float val1, float val2, float val3,
                      __local volatile float* smem1,
                      __local volatile float* smem2,
                      __local volatile float* smem3, int tid) {
    smem1[tid] = val1;
    smem2[tid] = val2;
    smem3[tid] = val3;
    barrier(CLK_LOCAL_MEM_FENCE);

    float a1, a2, a3;

    if (tid < 16)
    {
        smem1[tid] += smem1[tid + 16];
        smem2[tid] += smem2[tid + 16];
        smem3[tid] += smem3[tid + 16];
#if WAVE_SIZE < 16
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid < 1)
    {
#endif
        local float8* m1 = (local float8*)smem1;
        local float8* m2 = (local float8*)smem2;
        local float8* m3 = (local float8*)smem3;
        float8 t1 = m1[0]+m1[1];
        float8 t2 = m2[0]+m2[1];
        float8 t3 = m3[0]+m3[1];
        float4 t14 = t1.lo + t1.hi;
        float4 t24 = t2.lo + t2.hi;
        float4 t34 = t3.lo + t3.hi;
        smem1[0] = t14.x+t14.y+t14.z+t14.w;
        smem2[0] = t24.x+t24.y+t24.z+t24.w;
        smem3[0] = t34.x+t34.y+t34.z+t34.w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}


#define MAD_X1(src, dst) \
    dst = mad((0 - src.s0 - src.s2), (3.0f), dst);                                \
    dst = mad(src.s1,  (-10.0f), dst);
#define MAD_X3(src, dst) \
    dst = mad((src.s0 + src.s2), 3.0f, dst);                                   \
    dst = mad(src.s1, 10.0f, dst);                                             \
    *A11 = mad(dst, dst, *A11);
#define MAD_Y1(src, dst) \
    dst = mad(((-src.s0) + src.s2), 3.0f, dst);
#define MAD_Y2(src, dst) \
    dst = mad(((-src.s0) + src.s2), 10.0f, dst);
#define MAD_Y3(src, dstx, dsty) \
    dsty = mad(((-src.s0) + src.s2), 3.0f, dsty);                              \
    *A12 = mad(dstx, dsty, *A12);                                              \
    *A22 = mad(dsty, dsty, *A22);
inline void ReadandComputeSGM(image2d_t I, float2 Point, int xsize,
                       float3* dx, float3* dy, float3* pixel,
                       float* A11, float* A12, float* A22) {
    int sublid = get_sub_group_local_id();
    for(int i = 0; i < GRIDSIZE; ++i) {
        for(int j = 0; j < GRIDSIZE; ++j) {
            int2 coordA = (int2) (mad(i, xsize, (Point.x - 0.5f + j)) * 4,
                                  mad(6, sublid, (Point.y - 0.5f)));
            float8 blockA = as_float8(intel_sub_group_block_read8(I, coordA));

            int step = 2 * i;
            int step2 = 2*i+1;
#if 1
            if(j == 0) {
                MAD_X1(blockA.s012, dx[step].s0);
                MAD_X1(blockA.s123, dx[step].s1);
                MAD_X1(blockA.s234, dx[step].s2);
                MAD_X1(blockA.s345, dx[step2].s0);
                MAD_X1(blockA.s456, dx[step2].s1);
                MAD_X1(blockA.s567, dx[step2].s2);

                MAD_Y1(blockA.s012, dy[step].s0);
                MAD_Y1(blockA.s123, dy[step].s1);
                MAD_Y1(blockA.s234, dy[step].s2);
                MAD_Y1(blockA.s345, dy[step2].s0);
                MAD_Y1(blockA.s456, dy[step2].s1);
                MAD_Y1(blockA.s567, dy[step2].s2);
            } else if(j == 1) {
                MAD_Y2(blockA.s012, dy[step].s0);
                MAD_Y2(blockA.s123, dy[step].s1);
                MAD_Y2(blockA.s234, dy[step].s2);
                MAD_Y2(blockA.s345, dy[step2].s0);
                MAD_Y2(blockA.s456, dy[step2].s1);
                MAD_Y2(blockA.s567, dy[step2].s2);

                pixel[step] = blockA.s123;
                pixel[step2] = blockA.s456;
            } else {
                MAD_X3(blockA.s012, dx[step].s0);
                MAD_X3(blockA.s123, dx[step].s1);
                MAD_X3(blockA.s234, dx[step].s2);
                MAD_X3(blockA.s345, dx[step2].s0);
                MAD_X3(blockA.s456, dx[step2].s1);
                MAD_X3(blockA.s567, dx[step2].s2);

                MAD_Y3(blockA.s012, dx[step].s0, dy[step].s0);
                MAD_Y3(blockA.s123, dx[step].s1, dy[step].s1);
                MAD_Y3(blockA.s234, dx[step].s2, dy[step].s2);
                MAD_Y3(blockA.s345, dx[step2].s0, dy[step2].s0);
                MAD_Y3(blockA.s456, dx[step2].s1, dy[step2].s1);
                MAD_Y3(blockA.s567, dx[step2].s2, dy[step2].s2);
            }
#endif
        }
    }
}

inline void ReduceIMV(float val1, float val2,
                      __local volatile float* smem1,
                      __local volatile float* smem2,
                      int tid) {
    smem1[tid] = val1;
    smem2[tid] = val2;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 16)
    {
        smem1[tid] += smem1[tid + 16];
        smem2[tid] += smem2[tid + 16];
#if WAVE_SIZE < 16
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid<1)
    {
#endif
        local float8* m1 = (local float8*)smem1;
        local float8* m2 = (local float8*)smem2;
        float8 t1 = m1[0]+m1[1];
        float8 t2 = m2[0]+m2[1];
        float4 t14 = t1.lo + t1.hi;
        float4 t24 = t2.lo + t2.hi;
        smem1[0] = t14.x+t14.y+t14.z+t14.w;
        smem2[0]= t24.x+t24.y+t24.z+t24.w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

inline void ReadandComputeIMV(image2d_t J, float2 Point,
                              float3* dx, float3* dy, float3* pixel,
                              float* b1, float* b2) {

    int sublid = get_sub_group_local_id();
    for(int i = 0; i < GRIDSIZE; ++i) {
        int2 coordA = (int2) (mad(i, 8, (Point.x + 1.5f)) * 4,
                              mad(6, sublid, (Point.y + 0.5f)));
        float8 blockA = as_float8(intel_sub_group_block_read8(J, coordA));

        int step = 2 * i;
        int step2 = 2*i+1;
        float3 diff1 = blockA.s123 - pixel[step];
        float3 diff2 = blockA.s456 - pixel[step2];

        *b1 = dot(dx[step], diff1);
        *b1 += dot(dx[step2], diff2);
        *b2 = dot(dy[step], diff1);
        *b2 += dot(dy[step2], diff2);
    }
}

inline void ReduceErr(float val1, __local volatile float* smem1, int tid)
{
    smem1[tid] = val1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 16)
    {
        smem1[tid] += smem1[tid + 16];
#if WAVE_SIZE < 16
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (tid<1)
    {
#endif
        local float8* m1 = (local float8*)smem1;
        float8 t1 = m1[0]+m1[1];
        float4 t14 = t1.lo + t1.hi;
        smem1[0] = t14.x+t14.y+t14.z+t14.w;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

inline void GetErr(image2d_t J, float2 Point,  float3* pixel, float* errval) {
    int sublid = get_sub_group_local_id();
    for(int i = 0; i < GRIDSIZE; ++i) {
        int2 coordA = (int2) (mad(i, 8, Point.x) * 4,
                              mad(6, sublid, (Point.y + 0.5f)));
        float8 blockA = as_float8(intel_sub_group_block_read8(J, coordA));

        int step = 2 * i;
        int step2 = mad24(i, 2, 1);
        float3 diff1 = blockA.s123 - pixel[step];
        float3 diff2 = blockA.s456 - pixel[step2];

        diff1 += diff2;
        *errval = dot(diff1, (float3) 1);
    }
}
#endif

__attribute__((reqd_work_group_size(LSx, LSy, 1)))
__kernel void pyrlkSparse(image2d_t I, image2d_t J,
                       __global const float2* prevPts, __global float2* nextPts,
                       __global uchar* status, __global float* err,
                       const int level, const int rows, const int cols,
                       int PATCH_X, int PATCH_Y,
                       int c_winSize_x, int c_winSize_y,
                       int c_iters, char calcErr) {
#if 1
    printf("hehe = %d", get_local_id(0));
}
#else

    __local float smem1[BUFFER];
    __local float smem2[BUFFER];
    __local float smem3[BUFFER];

    int xlid = get_local_id(0);
    int ylid = get_local_id(1);
    int gid = get_group_id(0);
    int xsize = get_local_size(0);
    int ysize = get_local_size(1);
    const int tid = mad24(ylid, xsize, xlid);

    float wx = (mad24(xsize, 2, xlid) < c_winSize_x) ? 1 : 0;
    float wy = (mad24(ysize, 2, ylid) < c_winSize_y) ? 1 : 0;

    float2 prevPt = prevPts[gid] / (float2)(1 << level);
    if (prevPt.x < 0 || prevPt.x >= cols || prevPt.y < 0 || prevPt.y >= rows)
    {
        if (tid == 0 && level == 0)
        {
            status[gid] = 0;
        }

        return;
    }
    float2 c_halfWin = (float2)((c_winSize_x - 1)>>1, (c_winSize_y - 1)>>1);
    prevPt -= c_halfWin;

    // extract the patch from the first image, compute covariation matrix of derivatives

    float A11 = 0;
    float A12 = 0;
    float A22 = 0;

    float3 pixel[6];
    float3 dx[6];
    float3 dy[6];

    ReadandComputeSGM(I, prevPt, xsize,
                      dx, dy, pixel,
                      &A11, &A12, &A22);
    ReduceSGM(A11, A12, A22, smem1, smem2, smem3, tid);
    A11 = smem1[0];
    A12 = smem2[0];
    A22 = smem3[0];
    barrier(CLK_LOCAL_MEM_FENCE);

#if 0
    __local float testmem[24][24];
    for(int j = 0; j < 3; ++j) {
            testmme[ylid*6][8*j] = pixel[2*j].s0;
            testmme[ylid*6+1][8*j] = pixel[2*j].s1;
            testmme[ylid*6+2][8*j] = pixel[2*j].s2;
            testmme[ylid*6+3][8*j] = pixel[2*j+1].s0;
            testmme[ylid*6+4][8*j] = pixel[2*j+1].s1;
            testmme[ylid*6+5][8*j] = pixel[2*j+1].s2;
    }
    if(xlid == 0 && ylid == 0) {
        for(int i = 0; i < 24; ++i) {
            printf("%f, ", testmem[0][i]);
        }
        printf(" \n");
    }
#endif
    float D = mad(A11, A22, - A12 * A12);
    if (D < 1.192092896e-07f)
    {
        if (tid == 0 && level == 0)
            status[gid] = 0;

        return;
    }

    A11 /= D;
    A12 /= D;
    A22 /= D;

    prevPt = mad(nextPts[gid], 2.0f, - c_halfWin);

    for (int k = 0; k < c_iters; ++k)
    {
        if (prevPt.x < -c_halfWin.x || prevPt.x >= cols || prevPt.y < -c_halfWin.y || prevPt.y >= rows)
        {
            if (tid == 0 && level == 0)
                status[gid] = 0;
            break;
        }
        float b1 = 0;
        float b2 = 0;

        ReadandComputeIMV(J, prevPt,
                          dx, dy, pixel,
                          &b1, &b2);

        ReduceIMV(b1, b2, smem1, smem2, tid);

        b1 = smem1[0];
        b2 = smem2[0];
        barrier(CLK_LOCAL_MEM_FENCE);

        float2 delta;
        delta.x = mad(A12, b2, - A22 * b1) * 32.0f;
        delta.y = mad(A12, b1, - A11 * b2) * 32.0f;
        prevPt += delta;

        if (fabs(delta.x) < THRESHOLD && fabs(delta.y) < THRESHOLD)
            break;
    }

    D = 0.0f;
    if (calcErr)
    {
        GetErr(J, prevPt,  pixel, &D);
        ReduceErr(D, smem1, tid);
    }

    if (tid == 0)
    {
        prevPt += c_halfWin;

        nextPts[gid] = prevPt;

        if (calcErr)
            err[gid] = smem1[0] / (float)(c_winSize_x * c_winSize_y);
    }
}
#endif
