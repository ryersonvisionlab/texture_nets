#include "caffe/layers/noise_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void NoiseDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  AddNoiseBlob();

  CHECK(data_) << "NoiseDataLayer needs to be initialized by calling Reset";

  top[0]->Reshape(batch_size_, channels_, spatial_size_, spatial_size_);
  top[0]->data()->set_gpu_data(data_);
}

template <typename Dtype>
void NoiseDataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

INSTANTIATE_LAYER_GPU_FUNCS(NoiseDataLayer);

}  // namespace caffe

