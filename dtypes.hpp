#ifndef _DTYPE_H_
#define _DTYPE_H_

#include "cfg.hpp"
#include <ap_int.h>
#include <cassert>
#include <cstdio>
//#include <hls_video.h>

template <class T> T reg(T in) {
#pragma HLS inline off
#pragma HLS INTERFACE ap_none port = return register
    return in;
}

// Index data type define
typedef ap_uint<5> peidx_t;
typedef ap_int<INPUT_IDX_BITS> dimidx_t;
typedef ap_uint<CHANNEL_IDX_BITS> cidx_t;
typedef ap_uint<WEIGHTS_OFFSET_BITS> widx_t;
typedef ap_uint<24> imidx_t;

typedef ap_uint<4> flt_idx;
typedef ap_uint<14> cacheline_idx_t;

// must remain <= 23 bits to fit into float
typedef ap_uint<DRAM_ADDR_BITS> memaddr_t;
// kernel size (=1 or =3)
typedef ap_uint<2> kernel_t;
// conv stride (=1 or =2)
typedef ap_uint<2> stride_t;
// filter element count
typedef ap_uint<4> numfilterelems_t; // either =1 or =9
// cache lines
typedef ap_uint<2> cacheline_t; // cache height = 3 or 4 lines
typedef ap_uint<24> imgdramoffset_t;
typedef ap_uint<24> imgcacheaddr_t;
typedef ap_uint<24> pixelperrow_t;
typedef ap_int<24> coordinate_t;

typedef ap_int<8> data8_t;
typedef ap_int<10> data10_t;
typedef ap_int<16> data16_t;
typedef ap_int<32> data32_t;

struct conv_t {
    dimidx_t h, w;
    cidx_t ic, oc;
    kernel_t kernel;
    stride_t stride;
    bool leaky;
    memaddr_t inputs;
    memaddr_t weights;
    memaddr_t outputs;
    memaddr_t bias;
    memaddr_t scale;

    conv_t(int h, int w, int ic, int oc, int k, int s = 1, bool leaky = false,
           unsigned int inputs = 0ul, unsigned int weights = 0ul,
           unsigned int outputs = 0ul, unsigned int bias = 0ul,
           unsigned int scale = 0ul)
        : h(h), w(w), ic(ic), oc(oc), kernel(k), stride(s), leaky(leaky),
          inputs(inputs), weights(weights), outputs(outputs), bias(bias),
          scale(scale) {}

    conv_t()
        : h(0), w(0), ic(0), oc(0), kernel(0), stride(0), leaky(false),
          inputs(0), weights(0), outputs(0), bias(0), scale(0) {}
};

#define LOG(...) printf("LOG %s:%d ", __FILE__, __LINE__), printf(__VA_ARGS__)

template <typename T, size_t burst>
void copy_dram(T* dst, volatile T* src, int n) {
#pragma HLS INLINE
    for (int i = 0; i < n; i += burst) {
        T* BRAM = &dst[i];
        volatile T* DRAM = &src[i];
        for (int c = 0; c < burst; c++) {
#pragma HLS PIPELINE
            BRAM[c] = DRAM[c];
        }
    }
}

#endif
