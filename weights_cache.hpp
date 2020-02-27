#ifndef _WEIGHTS_CACHE_H_
#define _WEIGHTS_CACHE_H_
#include "dtypes.hpp"

namespace WeightsCache {
extern cidx_t align;
extern data8_t WBRAM[2048][N_PE][9];
void getIndex(const cidx_t oc, const widx_t ic_offset,
		cacheline_idx_t &line, peidx_t &peid);
// calculate input channel offset
widx_t getInputChannelOffset(const cidx_t ic);
// load weights to BRAM
void loadWeights(volatile const data8_t *SHARED_DRAM);
// fetch 9 weights from BRAM
void fetch9Weights(widx_t ic_offset, cidx_t oc, data8_t weights[9]);
};

#endif
