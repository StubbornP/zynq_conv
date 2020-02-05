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
	memaddr_t scale = conv_cfg.scale;
	memaddr_t bias = conv_cfg.bias;
	for (cidx_t i=0; i<oc; i++){
#pragma HLS LOOP_TRIPCOUNT MIN=16 AVG=80 MAX=520
		data32_t s = SHARED_DRAM[scale + i];
		SCALE[i] = s;
		LOG("loading post-process scale[%d]=%d\n", (int)i, (int)s);
	}
	for (cidx_t i=0; i<oc; i++){
#pragma HLS LOOP_TRIPCOUNT MIN=16 AVG=80 MAX=520
		data32_t b = SHARED_DRAM[bias + i];
		BIAS[i] = b;
		LOG("loading post-process bias[%d] = %d\n", (int)i, (int)b);
	}
}
data8_t postProcess(cidx_t co, data32_t out) {
#pragma HLS INLINE
#pragma HLS PIPELINE
	const conv_t &conv_cfg = ConfigBoard::getConv();
	data8_t ret;
	data32_t temp, scale, bias;

	if (conv_cfg.leaky && out < 0) {
		temp = out / 10;
	} else {
		temp = out;
	}

	scale = SCALE[co];
	bias = BIAS[co];

	temp = (temp + bias) / scale;
#pragma HLS RESOURCE variable=temp core=MulnS latency=3
	ret = data8_t(temp);
	return ret;
}
};
