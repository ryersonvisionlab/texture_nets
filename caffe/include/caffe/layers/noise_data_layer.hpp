#ifndef CAFFE_NOISE_DATA_LAYER_HPP_
#define CAFFE_NOISE_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

/**
 * @brief Provides generated noise data to the Net.
 *
 */
template <typename Dtype>
class NoiseDataLayer : public BaseDataLayer<Dtype> {
 
 public:
  explicit NoiseDataLayer(const LayerParameter& param)
      : BaseDataLayer<Dtype>(param) {}
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "NoiseData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  virtual void AddNoiseBlob();

  void Reset(Dtype* data);

  int batch_size() { return batch_size_; }
  int channels() { return channels_; }
  int spatial_size() { return spatial_size_; }
  string& distribution() { return distribution_; }
  float mu() { return mu_; }
  float sigma() { return sigma_; }
  float min() { return min_; }
  float max() { return max_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  int batch_size_, channels_, spatial_size_, size_;
  string distribution_;
  float mu_, sigma_, min_, max_;
  Dtype* data_;
  Blob<Dtype> added_data_;
};

}  // namespace caffe

#endif  // CAFFE_NOISE_DATA_LAYER_HPP_
