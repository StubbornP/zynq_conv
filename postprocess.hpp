#ifndef _POST_PROCESS_H_
#define _POST_PROCESS_H_

#include "dtypes.hpp"

namespace PostProcess {
void loadParams(volatile const data32_t *SHARED_DRAM);
data8_t postProcess(cidx_t co, data32_t out);
}

#endif
