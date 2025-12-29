/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#include "texture.cuh"
void sort (int numElems, unsigned int maxValue, wrap::cuda::SurfaceObject<unsigned int> &dkeys, wrap::cuda::SurfaceObject<unsigned int> &dvalues, int surfW, int surfH);
