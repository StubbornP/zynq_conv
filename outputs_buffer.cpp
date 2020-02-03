#include "outputs_buffer.hpp"
#include "config_board.hpp"

namespace OutputsBuffer {
data32_t OBRAM[MAX_CHANNEL_OUT] = {0};
memaddr_t stride, DRAMAddr;

void setup() {
	const conv_t& conv_cfg = ConfigBoard::getConv();
	stride = conv_cfg.w * conv_cfg.oc;
}

void setDRAMAddress(dimidx_t h, dimidx_t w) {
#pragma HLS INLINE
#pragma HLS PIPELINE
	const conv_t& conv_cfg = ConfigBoard::getConv();
    // Calculate Output Memory Address
	memaddr_t px_off = h * stride + w * conv_cfg.oc;
#pragma HLS RESOURCE variable=px_off core=MulnS latency=2
    DRAMAddr = conv_cfg.outputs + px_off;
    LOG("OutputsBuffer: set DRAM address: %x:%x\n", (int)conv_cfg.outputs, (int)DRAMAddr);
}

data32_t getOutputChannel(cidx_t co) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=OBRAM cyclic factor=N_PE
#pragma HLS RESOURCE variable=OBRAM core=RAM_T2P_BRAM latency=2
	return OBRAM[co];
}
void putOutputChannel(cidx_t co, data32_t val) {
#pragma HLS INLINE
#pragma HLS FUNCTION_INSTANTIATE variable=co
	OBRAM[co] = val;
}

void accOutputChannel(cidx_t co, data32_t val) {
#pragma HLS INLINE
#pragma HLS FUNCTION_INSTANTIATE variable=co
	data32_t temp = getOutputChannel(co);
	data32_t new_val = temp + val;
	putOutputChannel(co, new_val);
}

void flushOutputChannel(volatile data8_t* SHARED_DRAM, cidx_t co, data8_t val) {
#pragma HLS INLINE
    SHARED_DRAM[DRAMAddr + co] = val;
    LOG("OutputsBuffer: set DRAM address: %x, co:%d, val: %d\n", (int)DRAMAddr, (int)co, (int)val);
}
}; // namespace OutputsBuffer
