#ifndef _PROCESS_ELEMENT_H_
#define _PROCESS_ELEMENT_H_

#include "process_element.hpp"
#include "config_board.hpp"
#include "inputs_cache.hpp"
#include "weights_cache.hpp"
#include "outputs_buffer.hpp"

namespace ProcessElement {
namespace Internal {

void macc(const data8_t in[16], const data16_t weights[16],
		data32_t result[16]) {
#pragma HLS INLINE
	for (int i=0; i<16; i++) {
#pragma HLS UNROLL
		result[i] += in[i] * weights[i];
	}
}

void processOC(cidx_t ci, data8_t inputs[16]) {
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
		data16_t weights[16];
		data32_t result[16];
#pragma HLS ARRAY_PARTITION variable=weights complete dim=0
#pragma HLS ARRAY_PARTITION variable=result complete dim=0
		WeightsCache::fetchWeights(ci_offset, co, weights);
		if (!clear) {
			OutputsBuffer::getOutputChannel(co, result);
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
    InputsCache::Index idx[16];
    InputsCache::get9Index(h, w, idx);
#pragma HLS ARRAY_PARTITION variable=idx complete dim=0

TOP_CI:
    for (cidx_t ci = 0; ci < conv_ic; ci++) {
#pragma HLS LOOP_TRIPCOUNT min = 8 max = 520 avg = 45
        data8_t inputs[16];
#pragma HLS ARRAY_PARTITION variable=inputs complete dim=0
    	InputsCache::fetchInputs(ci, idx, inputs);
    	Internal::processOC(ci, inputs);
    }
}
};

#endif
