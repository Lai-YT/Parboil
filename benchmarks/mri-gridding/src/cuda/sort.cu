/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#include <cuda.h>
#include <stdio.h>

#include "scanLargeArray.h"
#include "texture.cuh"

#define UINT32_MAX 4294967295
#define BITS 4
#define LNB 4

#define SORT_BS 256

#define CONFLICT_FREE_OFFSET(index) ((index) >> LNB + (index) >> (2*LNB))
#define BLOCK_P_OFFSET (4*SORT_BS+1+(4*SORT_BS+1)/16+(4*SORT_BS+1)/64)

__device__ void scan (unsigned int s_data[BLOCK_P_OFFSET]){
  unsigned int thid = threadIdx.x;

  __syncthreads();

  s_data[2*thid+1+CONFLICT_FREE_OFFSET(2*thid+1)] += s_data[2*thid+CONFLICT_FREE_OFFSET(2*thid)];
  s_data[2*(blockDim.x+thid)+1+CONFLICT_FREE_OFFSET(2*(blockDim.x+thid)+1)] += s_data[2*(blockDim.x+thid)+CONFLICT_FREE_OFFSET(2*(blockDim.x+thid))];

  unsigned int stride = 2;
  for (unsigned int d = blockDim.x; d > 0; d >>= 1)
  {
    __syncthreads();

    if (thid < d)
    {
      unsigned int i  = 2*stride*thid;
      unsigned int ai = i + stride - 1;
      unsigned int bi = ai + stride;

      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      s_data[bi] += s_data[ai];
    }

    stride *= 2;
  }

  if (thid == 0){
    unsigned int last = 4*blockDim.x-1;
    last += CONFLICT_FREE_OFFSET(last);
    s_data[4*blockDim.x+CONFLICT_FREE_OFFSET(4*blockDim.x)] = s_data[last];
    s_data[last] = 0;
  }

  for (unsigned int d = 1; d <= blockDim.x; d *= 2)
  {
    stride >>= 1;

    __syncthreads();

    if (thid < d)
    {
      unsigned int i  = 2*stride*thid;
      unsigned int ai = i + stride - 1;
      unsigned int bi = ai + stride;

      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      unsigned int t  = s_data[ai];
      s_data[ai] = s_data[bi];
      s_data[bi] += t;
    }
  }
  __syncthreads();

  unsigned int temp = s_data[2*thid+CONFLICT_FREE_OFFSET(2*thid)];
  s_data[2*thid+CONFLICT_FREE_OFFSET(2*thid)] = s_data[2*thid+1+CONFLICT_FREE_OFFSET(2*thid+1)];
  s_data[2*thid+1+CONFLICT_FREE_OFFSET(2*thid+1)] += temp;

  unsigned int temp2 = s_data[2*(blockDim.x+thid)+CONFLICT_FREE_OFFSET(2*(blockDim.x+thid))];
  s_data[2*(blockDim.x+thid)+CONFLICT_FREE_OFFSET(2*(blockDim.x+thid))] = s_data[2*(blockDim.x+thid)+1+CONFLICT_FREE_OFFSET(2*(blockDim.x+thid)+1)];
  s_data[2*(blockDim.x+thid)+1+CONFLICT_FREE_OFFSET(2*(blockDim.x+thid)+1)] += temp2;

  __syncthreads();
}

__global__ static void splitSort(int numElems, int iter, cudaSurfaceObject_t keys_surf, cudaSurfaceObject_t values_surf, unsigned int* histo)
{
    __shared__ unsigned int flags[BLOCK_P_OFFSET];
    __shared__ unsigned int histo_s[1<<BITS];

    const unsigned int tid = threadIdx.x;
  const unsigned int local_idx = 4*threadIdx.x;
  const unsigned int gid = blockIdx.x*4*SORT_BS + local_idx;

    // Copy input to shared mem. Assumes input is always even numbered
    uint4 lkey = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    uint4 lvalue;
    if (gid < numElems){
      unsigned int k0,k1,k2,k3;
      unsigned int v0,v1,v2,v3;
      surf2Dread(&k0, keys_surf, local_idx * sizeof(unsigned int), blockIdx.x);
      surf2Dread(&k1, keys_surf, (local_idx+1) * sizeof(unsigned int), blockIdx.x);
      surf2Dread(&k2, keys_surf, (local_idx+2) * sizeof(unsigned int), blockIdx.x);
      surf2Dread(&k3, keys_surf, (local_idx+3) * sizeof(unsigned int), blockIdx.x);
      surf2Dread(&v0, values_surf, local_idx * sizeof(unsigned int), blockIdx.x);
      surf2Dread(&v1, values_surf, (local_idx+1) * sizeof(unsigned int), blockIdx.x);
      surf2Dread(&v2, values_surf, (local_idx+2) * sizeof(unsigned int), blockIdx.x);
      surf2Dread(&v3, values_surf, (local_idx+3) * sizeof(unsigned int), blockIdx.x);
      lkey.x = k0; lkey.y = k1; lkey.z = k2; lkey.w = k3;
      lvalue.x = v0; lvalue.y = v1; lvalue.z = v2; lvalue.w = v3;
    }

    if(tid < (1<<BITS)){
      histo_s[tid] = 0;
    }
    __syncthreads();

    atomicAdd(histo_s+((lkey.x&((1<<(BITS*(iter+1)))-1))>>(BITS*iter)),1);
    atomicAdd(histo_s+((lkey.y&((1<<(BITS*(iter+1)))-1))>>(BITS*iter)),1);
    atomicAdd(histo_s+((lkey.z&((1<<(BITS*(iter+1)))-1))>>(BITS*iter)),1);
    atomicAdd(histo_s+((lkey.w&((1<<(BITS*(iter+1)))-1))>>(BITS*iter)),1);

    uint4 index = {4*tid, 4*tid+1, 4*tid+2, 4*tid+3};

    for (int i=BITS*iter; i<BITS*(iter+1);i++){
      const uint4 flag = {(lkey.x>>i)&0x1,(lkey.y>>i)&0x1,(lkey.z>>i)&0x1,(lkey.w>>i)&0x1};

      flags[index.x+CONFLICT_FREE_OFFSET(index.x)] = 1<<(16*flag.x);
      flags[index.y+CONFLICT_FREE_OFFSET(index.y)] = 1<<(16*flag.y);
      flags[index.z+CONFLICT_FREE_OFFSET(index.z)] = 1<<(16*flag.z);
      flags[index.w+CONFLICT_FREE_OFFSET(index.w)] = 1<<(16*flag.w);

      scan (flags);

      index.x = (flags[index.x+CONFLICT_FREE_OFFSET(index.x)]>>(16*flag.x))&0xFFFF;
      index.y = (flags[index.y+CONFLICT_FREE_OFFSET(index.y)]>>(16*flag.y))&0xFFFF;
      index.z = (flags[index.z+CONFLICT_FREE_OFFSET(index.z)]>>(16*flag.z))&0xFFFF;
      index.w = (flags[index.w+CONFLICT_FREE_OFFSET(index.w)]>>(16*flag.w))&0xFFFF;

      unsigned short offset = flags[4*blockDim.x+CONFLICT_FREE_OFFSET(4*blockDim.x)]&0xFFFF;
      index.x += (flag.x) ? offset : 0;
      index.y += (flag.y) ? offset : 0;
      index.z += (flag.z) ? offset : 0;
      index.w += (flag.w) ? offset : 0;

      __syncthreads();
    }

    // Write result.
    if (gid < numElems){
      surf2Dwrite(lkey.x, keys_surf, index.x * sizeof(unsigned int), blockIdx.x);
      surf2Dwrite(lkey.y, keys_surf, index.y * sizeof(unsigned int), blockIdx.x);
      surf2Dwrite(lkey.z, keys_surf, index.z * sizeof(unsigned int), blockIdx.x);
      surf2Dwrite(lkey.w, keys_surf, index.w * sizeof(unsigned int), blockIdx.x);

      surf2Dwrite(lvalue.x, values_surf, index.x * sizeof(unsigned int), blockIdx.x);
      surf2Dwrite(lvalue.y, values_surf, index.y * sizeof(unsigned int), blockIdx.x);
      surf2Dwrite(lvalue.z, values_surf, index.z * sizeof(unsigned int), blockIdx.x);
      surf2Dwrite(lvalue.w, values_surf, index.w * sizeof(unsigned int), blockIdx.x);
    }
    if (tid < (1<<BITS)){
      histo[gridDim.x*threadIdx.x+blockIdx.x] = histo_s[tid];
    }
}

__global__ void splitRearrange (int numElems, int iter, cudaSurfaceObject_t keys_i_surf, cudaSurfaceObject_t keys_o_surf, cudaSurfaceObject_t values_i_surf, cudaSurfaceObject_t values_o_surf, unsigned int* histo){
  __shared__ unsigned int histo_s[(1<<BITS)];
  __shared__ unsigned int array_s[4*SORT_BS];
  int local_idx = 4*threadIdx.x;
  int index = blockIdx.x*4*SORT_BS + local_idx;

  if (threadIdx.x < (1<<BITS)){
    histo_s[threadIdx.x] = histo[gridDim.x*threadIdx.x+blockIdx.x];
  }

  uint4 mine, value;
  if (index < numElems){
    unsigned int k0,k1,k2,k3;
    unsigned int v0,v1,v2,v3;
    surf2Dread(&k0, keys_i_surf, local_idx * sizeof(unsigned int), blockIdx.x);
    surf2Dread(&k1, keys_i_surf, (local_idx+1) * sizeof(unsigned int), blockIdx.x);
    surf2Dread(&k2, keys_i_surf, (local_idx+2) * sizeof(unsigned int), blockIdx.x);
    surf2Dread(&k3, keys_i_surf, (local_idx+3) * sizeof(unsigned int), blockIdx.x);
    surf2Dread(&v0, values_i_surf, local_idx * sizeof(unsigned int), blockIdx.x);
    surf2Dread(&v1, values_i_surf, (local_idx+1) * sizeof(unsigned int), blockIdx.x);
    surf2Dread(&v2, values_i_surf, (local_idx+2) * sizeof(unsigned int), blockIdx.x);
    surf2Dread(&v3, values_i_surf, (local_idx+3) * sizeof(unsigned int), blockIdx.x);
    mine.x = k0; mine.y = k1; mine.z = k2; mine.w = k3;
    value.x = v0; value.y = v1; value.z = v2; value.w = v3;
  } else {
    mine.x = UINT32_MAX;
    mine.y = UINT32_MAX;
    mine.z = UINT32_MAX;
    mine.w = UINT32_MAX;
  }
  uint4 masks = {(mine.x&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine.y&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine.z&((1<<(BITS*(iter+1)))-1))>>(BITS*iter),
                 (mine.w&((1<<(BITS*(iter+1)))-1))>>(BITS*iter)};

  ((uint4*)array_s)[threadIdx.x] = masks;
  __syncthreads();

  uint4 new_index = {histo_s[masks.x],histo_s[masks.y],histo_s[masks.z],histo_s[masks.w]};

  int i = 4*threadIdx.x-1;
  while (i >= 0){
    if (array_s[i] == masks.x){
      new_index.x++;
      i--;
    } else {
      break;
    }
  }

  new_index.y = (masks.y == masks.x) ? new_index.x+1 : new_index.y;
  new_index.z = (masks.z == masks.y) ? new_index.y+1 : new_index.z;
  new_index.w = (masks.w == masks.z) ? new_index.z+1 : new_index.w;

  if (index < numElems){
#define BLK(val) ((val) / (4*SORT_BS))
#define OFF(val) ((val) % (4*SORT_BS))

    surf2Dwrite(mine.x, keys_o_surf, OFF(new_index.x) * sizeof(unsigned int), BLK(new_index.x));
    surf2Dwrite(value.x, values_o_surf, OFF(new_index.x) * sizeof(unsigned int), BLK(new_index.x));

    surf2Dwrite(mine.y, keys_o_surf, OFF(new_index.y) * sizeof(unsigned int), BLK(new_index.y));
    surf2Dwrite(value.y, values_o_surf, OFF(new_index.y) * sizeof(unsigned int), BLK(new_index.y));

    surf2Dwrite(mine.z, keys_o_surf, OFF(new_index.z) * sizeof(unsigned int), BLK(new_index.z));
    surf2Dwrite(value.z, values_o_surf, OFF(new_index.z) * sizeof(unsigned int), BLK(new_index.z));

    surf2Dwrite(mine.w, keys_o_surf, OFF(new_index.w) * sizeof(unsigned int), BLK(new_index.w));
    surf2Dwrite(value.w, values_o_surf, OFF(new_index.w) * sizeof(unsigned int), BLK(new_index.w));
#undef BLK
#undef OFF
  }
}

void sort (int numElems, unsigned int max_value, wrap::cuda::SurfaceObject<unsigned int> &dkeys, wrap::cuda::SurfaceObject<unsigned int> &dvalues, int surfW, int surfH){
  dim3 grid (surfH);
  dim3 block (SORT_BS);

  unsigned int iterations = 0;
  while(max_value > 0){
    max_value >>= BITS;
    iterations++;
  }

  unsigned int *dhisto;
  wrap::cuda::SurfaceObject<unsigned int> keys_o_surf, values_o_surf;

  cudaMalloc((void**)&dhisto, (1<<BITS)*grid.x*sizeof(unsigned int));
  if (wrap::cuda::malloc2DSurfaceObject<unsigned int>(&keys_o_surf, surfW, surfH) != cudaSuccess){
    printf("Failed allocating keys_o surface in sort\n");
    return;
  }
  if (wrap::cuda::malloc2DSurfaceObject<unsigned int>(&values_o_surf, surfW, surfH) != cudaSuccess){
    printf("Failed allocating values_o surface in sort\n");
    return;
  }

  for (int i=0; i<iterations; i++){
    splitSort<<<grid,block>>>(numElems, i, dkeys.surf, dvalues.surf, dhisto);

    scanLargeArray(grid.x*(1<<BITS), dhisto);

    splitRearrange<<<grid,block>>>(numElems, i, dkeys.surf, keys_o_surf.surf, dvalues.surf, values_o_surf.surf, dhisto);

    // swap surface objects
    { auto tmp = dkeys; dkeys = keys_o_surf; keys_o_surf = tmp; }
    { auto tmp = dvalues; dvalues = values_o_surf; values_o_surf = tmp; }
  }

  wrap::cuda::freeSurfaceObject<unsigned int>(&keys_o_surf);
  wrap::cuda::freeSurfaceObject<unsigned int>(&values_o_surf);
  cudaFree(dhisto);
}

/* vim: set ts=2 sw=2 sts=2 et ai: */
