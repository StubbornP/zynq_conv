#ifndef _INPUTS_CACHE_H_
#define _INPUTS_CACHE_H_
#include "dtypes.hpp"

namespace InputsCache {
namespace Internal {
void loadInputChannel(volatile data8_t *SHARED_DRAM);
};
using Internal::loadInputChannel;
void reset();
void setDRAMRow(dimidx_t row);
void loadRow(volatile data8_t* SHARED_DRAM);
imidx_t getRowOffset(const dimidx_t h);
data8_t fetchCachePixel(const dimidx_t h, const dimidx_t w,
                       const imidx_t row_offset, const cidx_t ci);
void fetchInputs(dimidx_t h, dimidx_t w, cidx_t ic, data8_t inputs[9]);
void inputsCacheTest(conv_t conv, volatile data8_t *SHARED_DRAM, data32_t cmd);
}; // namespace InputsCache
#endif
