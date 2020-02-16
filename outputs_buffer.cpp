#include "outputs_buffer.hpp"
#include "config_board.hpp"
#include "postprocess.hpp"

namespace OutputsBuffer {
data32_t OBRAM[512][4];
dimidx_t OW;
imidx_t DRAMAddr[4];

void setup() {
#pragma HLS ARRAY_PARTITION variable=OBRAM cyclic dim=1 factor=N_PE
#pragma HLS ARRAY_PARTITION variable=OBRAM complete dim=2
#pragma HLS RESOURCE variable=OBRAM core=RAM_T2P_BRAM latency=1
	const conv_t conv_cfg = ConfigBoard::getConv();
	const dimidx_t w = conv_cfg.w;
	OW = (conv_cfg.stride==2)?dimidx_t(w+1/2):w;
}

void setDRAMAddress(dimidx_t oh, dimidx_t ow) {
#pragma HLS INLINE
#pragma HLS PIPELINE
	const conv_t conv_cfg = ConfigBoard::getConv();
	const memaddr_t outputs = conv_cfg.outputs;
	const cidx_t oc = conv_cfg.oc;
    // Calculate Output Memory Address
	imidx_t px_base = oh * OW + ow;
//	px_base = outputs + oh * WStride + ow * oc;
    DRAMAddr[0] = px_base;
    DRAMAddr[1] = px_base + 1;
    DRAMAddr[2] = px_base + OW;
    DRAMAddr[3] = px_base + OW + 1;
    LOG("OutputsBuffer: set DRAM address: %x:%x\n", (int)conv_cfg.outputs, (int)DRAMAddr[0]);
}

void getOutputChannel(cidx_t co, bool clear, data32_t out[4]) {
#pragma HLS INLINE
#pragma HLS FUNCTION_INSTANTIATE variable=co
	for (int i=0; i<4; i++) {
#pragma HLS PIPELINE
		if (clear) {
			out[i] = 0;
		} else {
			out[i] = OBRAM[co][i];
		}
	}
}
void putOutputChannel(cidx_t co, data32_t out[4]) {
#pragma HLS INLINE
#pragma HLS FUNCTION_INSTANTIATE variable=co
	for (int i=0; i<4; i++) {
#pragma HLS PIPELINE
		OBRAM[co][i] = out[i];
	}
}

void flushOutputChannel(volatile data8_t* SHARED_DRAM) {
#pragma HLS INLINE
    const conv_t conv_cfg = ConfigBoard::getConv();
    const cidx_t OC = conv_cfg.oc;
	const memaddr_t outputs = conv_cfg.outputs;
    for (int i=0; i<4; i++) {
    	memaddr_t addr = outputs + DRAMAddr[i] * OC;
    	volatile data8_t *Out = &SHARED_DRAM[addr];
        for (cidx_t co=0; co<OC; co++) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 520 avg = 45
        	data32_t bram = OBRAM[co][i] / 4;
         	Out[co] = PostProcess::postProcess(co, bram);
        	LOG("OutputsBuffer: set DRAM address: %x, co:%d, val: %d\n", (int)DRAMAddr[0], (int)co, (int)bram);
        }
    }
}
}; // namespace OutputsBuffer
