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
	data32_t out[4];
#pragma HLS ARRAY_PARTITION variable=out complete dim=0

	for (int i=0; i<4; i++) {
#pragma HLS UNROLL
		out[i] = in[i] * weights[i];
	}
	for (int i=0; i<4; i++) {
#pragma HLS UNROLL
		out[i] += in[4+i] * weights[4+i];
	}
	res = result + in[8] * weights[8];
	for (int i=0; i<2; i++) {
#pragma HLS UNROLL
		out[i] += out[2+i];
	}
	res += out[0] + out[1];
	result = res;
}

void processOC(dimidx_t h, dimidx_t w, cidx_t ci, const data8_t in[9], bool clear) {
	const conv_t &conv_cfg = ConfigBoard::getConv();
	const cidx_t oc = conv_cfg.oc;
#pragma HLS INLINE OFF
    const widx_t ci_offset = ci * WeightsCache::align;
L_PROCESS_OC:
	for (cidx_t co=0; co<oc; co++) {
#pragma HLS LOOP_TRIPCOUNT min = 32 max = 520 avg = 150
#pragma HLS UNROLL factor=N_PE
#pragma HLS PIPELINE II=1
		data32_t result(0);
		data16_t weights[9];
#pragma HLS ARRAY_PARTITION variable=weights complete dim=0
		WeightsCache::fetch9Weights(ci_offset, co, weights);
		if (!clear) {
			result += OutputsBuffer::getOutputChannel(co);
		}
		macc(in, weights, result);
		OutputsBuffer::putOutputChannel(co, result);
	}
}
}; // namespace Internal

void processIC(dimidx_t h, dimidx_t w, cidx_t ci) {
#pragma HLS INLINE
#pragma HLS DATAFLOW
#pragma HLS FUNCTION_INSTANTIATE variable=ci
	data8_t in[9];
#pragma HLS ARRAY_PARTITION variable=in complete dim=0
	InputsCache::fetchInputs(h, w, ci, in);
	Internal::processOC(h, w, ci, in, ci==0);
}
};

#endif
