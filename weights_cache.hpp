#ifndef _WEIGHTS_CACHE_H_
#define _WEIGHTS_CACHE_H_
#include "dtypes.hpp"

namespace WeightsCache {
extern cidx_t align;
extern data16_t WBRAM[1024][N_PE][16];
void getIndex(const cidx_t oc, const widx_t ic_offset,
		cacheline_idx_t &line, peidx_t &peid);
// calculate input channel offset
widx_t getInputChannelOffset(const cidx_t ic);
// load weights to BRAM
void loadWeights(volatile const data16_t *SHARED_DRAM);
// fetch 9 weights from BRAM
void fetchWeights(widx_t ic_offset, cidx_t oc, data16_t weights[16]);
// unit test
void weightsCacheTest(conv_t conv, volatile data16_t *SHARED_DRAM, data32_t cmd);
};

#endif
