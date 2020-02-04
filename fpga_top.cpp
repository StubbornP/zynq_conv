#include "fpga_top.hpp"
#include "config_board.hpp"
#include "inputs_cache.hpp"
#include "outputs_buffer.hpp"
#include "postprocess.hpp"
#include "process_element.hpp"
#include "weights_cache.hpp"

//void fpga_top(conv_t conv, data32_t cmd, volatile data8_t* SHM8_DRAM,
//              volatile data16_t* SHM16_DRAM, volatile data32_t* SHM32_DRAM) {
//#pragma HLS INTERFACE m_axi depth=DRAM_DEPTH port=SHM8_DRAM offset=0 bundle=data_bus register
//#pragma HLS INTERFACE m_axi depth=DRAM_DEPTH port=SHM16_DRAM offset=0 bundle=data_bus register
//#pragma HLS INTERFACE m_axi depth=DRAM_DEPTH port=SHM32_DRAM offset=0 bundle=data_bus register
//
//#pragma HLS INTERFACE s_axilite port = conv bundle = ctrl_bus register
//#pragma HLS INTERFACE s_axilite port = cmd bundle = ctrl_bus register
//#pragma HLS INTERFACE s_axilite port = return bundle = ctrl_bus register
//
//    ConfigBoard::setConv(conv);
//    const conv_t& conv_cfg = ConfigBoard::getConv();
//
//    InputsCache::reset();
//    OutputsBuffer::setup();
//    WeightsCache::loadWeights(SHM16_DRAM);
//    PostProcess::loadParams(SHM32_DRAM);
//
//    InputsCache::loadRow(SHM8_DRAM);          // preload 2 pixels, 1 pad
//    InputsCache::loadInputChannel(SHM8_DRAM); // 1 pixel per width
//
//    const dimidx_t conv_h = conv_cfg.h;
//    const dimidx_t conv_w = conv_cfg.w;
//    const cidx_t conv_ic = conv_cfg.ic;
//    const cidx_t conv_oc = conv_cfg.oc;
//
//    dimidx_t h, w;
//
//TOP_H:
//    for (h = 0; h < conv_h; h++) {
//#pragma HLS LOOP_TRIPCOUNT min = 8 max = 416 avg = 45
//    TOP_W:
//        for (w = 0; w < conv_w; w++) {
//#pragma HLS LOOP_TRIPCOUNT min = 8 max = 416 avg = 45
//
//            cidx_t ci, co;
//            dimidx_t oh, ow;
//
//            oh = h, ow = w;
//
//            if (ConfigBoard::is1x1Conv())
//                continue;
//
//            if (ConfigBoard::is3x3S2Conv()) {
//                oh = h / 2, ow = w / 2;
//                if (h % 2 | w % 2)
//                    continue;
//            }
//
//            OutputsBuffer::setDRAMAddress(oh, ow);
//
//            InputsCache::loadInputChannel(SHM8_DRAM);
//        TOP_CI:
//            for (ci = 0; ci < conv_ic; ci++) {
//#pragma HLS LOOP_TRIPCOUNT min = 8 max = 650 avg = 45
//                ProcessElement::processIC(h, w, ci);
//            }
//
//            for (co = 0; co < conv_oc; co++) {
//#pragma HLS LOOP_TRIPCOUNT min = 8 max = 650 avg = 45
//#pragma HLS pipeline II = 1
//                data32_t res = OutputsBuffer::getOutputChannel(co);
//                data8_t out = PostProcess::postProcess(co, res);
//                OutputsBuffer::flushOutputChannel(SHM8_DRAM, co, out);
//                LOG("writing back Out[%d, %d, %d] res: %d, out: %d\n", (int)h,
//                    (int)w, (int)co, (int)res, (int)out);
//            }
//        }
//    }
//}

void fpga_top(conv_t conv, data32_t cmd,
		volatile data16_t* SHM16_DRAM) {
#pragma HLS INTERFACE m_axi depth=DRAM_DEPTH port=SHM16_DRAM offset=0 bundle=data_bus register max_read_burst_length=9*N_PE

#pragma HLS INTERFACE s_axilite port = conv bundle = ctrl_bus register
#pragma HLS INTERFACE s_axilite port = cmd bundle = ctrl_bus register
#pragma HLS INTERFACE s_axilite port = return bundle = ctrl_bus register

    ConfigBoard::setConv(conv);
    const conv_t& conv_cfg = ConfigBoard::getConv();
    WeightsCache::weightsCacheTest(conv_cfg, SHM16_DRAM, cmd);
//    	InputsCache::inputsCacheTest(conv, SHARED_DRAM, cmd);
}


