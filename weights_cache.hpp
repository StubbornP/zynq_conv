#ifndef _WEIGHTS_CACHE_H_
#define _WEIGHTS_CACHE_H_
#include "dtypes.hpp"

namespace WeightsCache {
// calculate input channel offset
widx_t getInputChannelOffset(const cidx_t ic);
// load weights to BRAM
void loadWeights(volatile data16_t *SHARED_DRAM);
// fetch 9 weights from BRAM
void fetch9Weights(widx_t ic_offset, cidx_t oc, data16_t weights[9]);
// unit test
void weightsCacheTest(conv_t conv, volatile data16_t *SHARED_DRAM, data32_t cmd);
};

#endif
