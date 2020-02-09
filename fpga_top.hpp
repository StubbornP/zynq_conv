#ifndef _FPGA_TOP_H_
#define _FPGA_TOP_H_

#include "dtypes.hpp"

// void fpga_top(conv_t layer, volatile data_t* SHARED_DRAM, data_t cmd);

// void fpga_top(conv_t conv, data32_t cmd, volatile data8_t* SHM8_DRAM,
//               volatile data16_t* SHM16_DRAM, volatile data32_t* SHM32_DRAM);

void fpga_top(conv_t conv, data32_t cmd,
		volatile data16_t* SHM16_DRAM);

#endif
