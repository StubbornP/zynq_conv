#include "config_board.hpp"

namespace ConfigBoard {
conv_t conv_cfg;
const conv_t getConv() {
#pragma HLS INLINE
    return conv_cfg;
}
void setConv(conv_t& cfg) {
#pragma HLS INLINE
    conv_cfg = cfg;
}
}; // namespace ConfigBoard
