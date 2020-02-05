#ifndef _CONFIG_BOARD_H_
#define _CONFIG_BOARD_H_

#include "dtypes.hpp"

namespace ConfigBoard {
const conv_t& getConv();
void setConv(conv_t& c);
bool is1x1Conv();
bool is3x3S2Conv();
}; // namespace ConfigBoard

#endif
