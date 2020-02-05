#include "config_board.hpp"

namespace ConfigBoard {
conv_t conv_cfg;
const conv_t& getConv() {
#pragma HLS INLINE
    return conv_cfg;
}
void setConv(conv_t& cfg) {
#pragma HLS INLINE
    conv_cfg = cfg;
}
bool is1x1Conv() {
#pragma HLS INLINE
	return conv_cfg.kernel == 1 && conv_cfg.stride == 1;
}
bool is3x3S2Conv() {
#pragma HLS INLINE
	return conv_cfg.kernel == 3 && conv_cfg.stride == 2;
}

}; // namespace ConfigBoard
