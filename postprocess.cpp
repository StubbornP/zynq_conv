#include "config_board.hpp"
#include "postprocess.hpp"

namespace PostProcess{
data32_t SCALE[MAX_CHANNEL_OUT];
data32_t BIAS[MAX_CHANNEL_OUT];

void loadParams(volatile data32_t *SHARED_DRAM) {
#pragma HLS INLINE
#pragma HLS PIPELINE
	const conv_t &conv_cfg = ConfigBoard::getConv();
	const cidx_t oc = conv_cfg.oc;
	volatile data32_t *SCALE_DRAM = &SHARED_DRAM[0];
	volatile data32_t *BIAS_DRAM = &SHARED_DRAM[oc];

	copy_dram<data32_t, 32>(SCALE, SCALE_DRAM, oc);
	copy_dram<data32_t, 32>(BIAS, BIAS_DRAM, oc);
//
//
//	for (cidx_t i=0; i<oc; i++){
//#pragma HLS LOOP_TRIPCOUNT MIN=16 AVG=80 MAX=520
//		data32_t s = SHARED_DRAM[scale + i];
//		SCALE[i] = s;
//		LOG("loading post-process scale[%d]=%d\n", (int)i, (int)s);
//	}
//	for (cidx_t i=0; i<oc; i++){
//#pragma HLS LOOP_TRIPCOUNT MIN=16 AVG=80 MAX=520
//		data32_t b = SHARED_DRAM[bias + i];
//		BIAS[i] = b;
//		LOG("loading post-process bias[%d] = %d\n", (int)i, (int)b);
//	}
}
data8_t postProcess(cidx_t co, data32_t out) {
#pragma HLS INLINE
#pragma HLS PIPELINE
	const conv_t &conv_cfg = ConfigBoard::getConv();
	data8_t ret;
	data32_t act, scale, bias, quant;

	scale = SCALE[co];
	bias = BIAS[co];
	// bn and quant
	quant = (out + bias) / scale;

	if (conv_cfg.leaky && quant < 0) {
		act = quant / 16;
	} else {
		act = quant;
	}

	if (act > data32_t(127)) {
		ret = data8_t(127);
	} else if (act<data32_t(-128)){
		ret = data8_t(-128);
	} else {
		ret = data8_t(act);
	}
	return ret;
}
};
