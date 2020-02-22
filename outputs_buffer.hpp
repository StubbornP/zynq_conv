#ifndef _OUTPUTS_BUFFER_H_
#define _OUTPUTS_BUFFER_H_

#include "dtypes.hpp"

namespace OutputsBuffer {
void setup();
void getCacheIdx(cidx_t co, cacheline_t line, cacheline_idx_t& idx);
void setDRAMAddress(dimidx_t h, dimidx_t w);
void getOutputChannel(cidx_t co, bool clear, data32_t out[4]);
void putOutputChannel(cidx_t co, data32_t out[4]);
void flushOutputChannel(volatile data8_t* SHARED_DRAM);
}; // namespace OutputsBuffer

#endif
