#ifndef _PROCESS_ELEMENT_H_
#define _PROCESS_ELEMENT_H_

#include "process_element.hpp"
#include "config_board.hpp"
#include "inputs_cache.hpp"
#include "weights_cache.hpp"
#include "outputs_buffer.hpp"

namespace ProcessElement {
namespace Internal {
// requant
// bn
// leaky relu

void macc(const data8_t in[9], const data16_t weights[9], data32_t &result) {
#pragma HLS INLINE
	data32_t res = 0;
	data32_t out[9];
#pragma HLS ARRAY_PARTITION variable=out complete dim=0
#pragma HLS RESOURCE variable=out core=MulnS latency=3
#pragma HLS RESOURCE variable=res core=AddSub_DSP latency=3
    res = result;
	for (int i=0; i<9; i++) {
#pragma HLS UNROLL
		out[i] = in[i] * weights[i];
	}
	for (int i=0; i<9; i++) {
#pragma HLS UNROLL
#pragma HLS PIPELINE
		res += out[i];
	}
	result = res;
}

void processOC(dimidx_t h, dimidx_t w, cidx_t ci_offset, const data8_t in[9], bool clear) {
	const conv_t &conv_cfg = ConfigBoard::getConv();
	const cidx_t oc = conv_cfg.oc;
#pragma HLS INLINE off
	for (cidx_t co=0; co<oc; co++) {
#pragma HLS LOOP_TRIPCOUNT min = 32 max = 560 avg = 258
#pragma HLS unroll factor=N_PE
#pragma HLS PIPELINE II=4
		data16_t weights[9];
		data32_t result;
#pragma HLS ARRAY_PARTITION variable=weights complete dim=0
		if (clear) {
			result = data32_t(0);
		} else {
			result = OutputsBuffer::getOutputChannel(co);
		}
		WeightsCache::fetch9Weights(ci_offset, co, weights);
		macc(in, weights, result);
		OutputsBuffer::putOutputChannel(co, result);
	}
}
}; // namespace Internal

void loadInput(dimidx_t h, dimidx_t w, cidx_t ci, widx_t &ci_offset, data8_t in[9]) {
#pragma HLS INLINE OFF
	InputsCache::fetchInputs(h, w, ci, in);
	ci_offset = WeightsCache::getInputChannelOffset(ci);
}

void processIC(dimidx_t h, dimidx_t w, cidx_t ci) {
#pragma HLS DATAFLOW
#pragma HLS INLINE OFF
#pragma HLS FUNCTION_INSTANTIATE variable=ci
	data8_t in[9];
#pragma HLS ARRAY_PARTITION variable=in complete dim=0
	widx_t ci_offset;
	loadInput(h, w, ci, ci_offset, in);
	Internal::processOC(h, w, ci_offset, in, ci==0);
}
};

#endif
