#include "outputs_buffer.hpp"
#include "config_board.hpp"
#include "postprocess.hpp"

namespace OutputsBuffer {
data32_t OBRAM[MAX_CHANNEL_OUT];
memaddr_t WStride, DRAMAddr;

void setup() {
#pragma HLS ARRAY_PARTITION variable=OBRAM cyclic factor=N_PE
#pragma HLS RESOURCE variable=OBRAM core=RAM_T2P_BRAM latency=1
	const conv_t& conv_cfg = ConfigBoard::getConv();
	const dimidx_t w = conv_cfg.w;
	dimidx_t ow = (conv_cfg.stride==2)?dimidx_t(w+1/2):w;
	WStride = ow * conv_cfg.oc;
}

void setDRAMAddress(dimidx_t oh, dimidx_t ow) {
#pragma HLS INLINE
#pragma HLS PIPELINE
	const conv_t& conv_cfg = ConfigBoard::getConv();
    // Calculate Output Memory Address
	memaddr_t px_off = oh * WStride + ow * conv_cfg.oc;
    DRAMAddr = conv_cfg.outputs + px_off;
    LOG("OutputsBuffer: set DRAM address: %x:%x\n", (int)conv_cfg.outputs, (int)DRAMAddr);
}

data32_t getOutputChannel(cidx_t co) {
#pragma HLS INLINE
#pragma HLS FUNCTION_INSTANTIATE variable=co
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

void flushOutputChannel(volatile data8_t* SHARED_DRAM) {
#pragma HLS INLINE
    const conv_t& conv_cfg = ConfigBoard::getConv();
    const cidx_t conv_oc = conv_cfg.oc;
    volatile data8_t *Out = &SHARED_DRAM[DRAMAddr];

//    data32_t *BRAM = &OBRAM[0];
//    const cidx_t burst = 32;
//
//	for (int co=0; co<conv_oc;) {
//#pragma HLS LOOP_TRIPCOUNT min = 8 max = 50 avg = 35
//		for (int c=0; c<burst; c++) {
//#pragma HLS PIPELINE
//			data32_t o = BRAM[c];
//	     	Out[c] = PostProcess::postProcess(co, o);
//	     	co ++;
//	    	LOG("OutputsBuffer: set DRAM address: %x, co:%d, val: %d\n", (int)DRAMAddr, (int)co, (int)o);
//		}
//		Out += burst;
//		BRAM += burst;
//	}

    for (cidx_t co=0; co<conv_oc; co++) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 520 avg = 45
    	data32_t bram = OBRAM[co];
     	Out[co] = PostProcess::postProcess(co, bram);
    	LOG("OutputsBuffer: set DRAM address: %x, co:%d, val: %d\n", (int)DRAMAddr, (int)co, (int)bram);

    }
}
}; // namespace OutputsBuffer
