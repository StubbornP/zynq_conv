#ifndef _PROCESS_ELEMENT_H_
#define _PROCESS_ELEMENT_H_

#include "process_element.hpp"
#include "config_board.hpp"
#include "inputs_cache.hpp"
#include "weights_cache.hpp"
#include "outputs_buffer.hpp"

namespace ProcessElement {
namespace Internal {

void macc(const data8_t in[9], const data16_t weights[9], data32_t &result) {
#pragma HLS INLINE
	data32_t res;
	data32_t out[9];
#pragma HLS ARRAY_PARTITION variable=out complete dim=0
	res = result;
	for (int i=0; i<4; i++) {
#pragma HLS UNROLL
		out[i] = in[i] * weights[i];
		out[i] += in[i+4] * weights[i+4];
		res += out[i];
	}
	result = res + in[8] * weights[8];
}

void processOC(cidx_t ci, data8_t inputs[9]) {
#pragma HLS INLINE
#pragma HLS FUNCTION_INSTANTIATE variable=ci
	const conv_t &conv_cfg = ConfigBoard::getConv();
	const cidx_t oc = conv_cfg.oc;
    const widx_t ci_offset = ci * WeightsCache::align;
    const bool clear = (ci==0);

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
			result = OutputsBuffer::getOutputChannel(co);
		}
		macc(inputs, weights, result);
		OutputsBuffer::putOutputChannel(co, result);
	}
}
}; // namespace Internal

void processIC(dimidx_t h, dimidx_t w) {
#pragma HLS INLINE
#pragma HLS PIPELINE
    const conv_t& conv_cfg = ConfigBoard::getConv();
    const cidx_t conv_ic = conv_cfg.ic;
    InputsCache::Index idx[9];
    InputsCache::get9Index(h, w, idx);
#pragma HLS ARRAY_PARTITION variable=idx complete dim=0

TOP_CI:
    for (cidx_t ci = 0; ci < conv_ic; ci++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 520 avg = 45
        data8_t inputs[9];
#pragma HLS ARRAY_PARTITION variable=inputs complete dim=0
    	InputsCache::fetchInputs(ci, idx, inputs);
    	Internal::processOC(ci, inputs);
    }
}
};

#endif
