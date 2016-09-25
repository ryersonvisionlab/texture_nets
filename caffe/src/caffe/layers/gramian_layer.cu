#include "caffe/layers/gramian_layer.hpp"

namespace caffe {

template <typename Dtype>
void GramianLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int batch = 0; batch < bottom[0]->shape(0); ++batch) {
    const Dtype* bottom_data = bottom[0]->gpu_data() + bottom[0]->offset(batch);
    Dtype* mutable_top_data = top[0]->mutable_gpu_data() + top[0]->offset(batch);
    caffe_gpu_gemm<Dtype>(
        CblasNoTrans, 
        CblasTrans, 
        bottom[0]->shape(1),
        bottom[0]->shape(1),
        bottom[0]->count(2),
        normalize_scale_,
        bottom_data,
        bottom_data,
        (Dtype)0.,
        mutable_top_data);
  }
}

template <typename Dtype>
void GramianLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
  for (int batch = 0; batch < bottom[0]->shape(0); ++batch) {
    Dtype* mutable_bottom_diff = bottom[0]->mutable_gpu_diff() + bottom[0]->offset(batch);
    const Dtype* bottom_data = bottom[0]->gpu_data() + bottom[0]->offset(batch);
    const Dtype* top_diff = top[0]->gpu_diff() + top[0]->offset(batch);
    // Input gradient
    if (propagate_down[0]) {
      caffe_gpu_gemm<Dtype>(
          CblasNoTrans,
          CblasNoTrans,
          bottom[0]->shape(1),
          bottom[0]->count(2),
          bottom[0]->shape(1),
          normalize_scale_,
          top_diff,
          bottom_data,
          (Dtype)0.,
          mutable_bottom_diff);
      caffe_gpu_gemm<Dtype>(
          CblasTrans,
          CblasNoTrans,
          bottom[0]->shape(1),
          bottom[0]->count(2),
          bottom[0]->shape(1),
          normalize_scale_,
          top_diff,
          bottom_data,
          (Dtype)1.,
          mutable_bottom_diff);
      // normalize gradient
      if (this->layer_param_.gramian_param().normalize_output()) {
        Dtype L1 = 1;
        caffe_gpu_asum<Dtype>(bottom[0]->count(1), mutable_bottom_diff, &L1);
        // to prevent divide by zero errors
        L1 += 1e-8;
        // divide by L1 norm
        caffe_gpu_div_scalar<Dtype>(
            bottom[0]->count(1),
            L1,
            mutable_bottom_diff);
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(GramianLayer);

}  // namespace caffe
