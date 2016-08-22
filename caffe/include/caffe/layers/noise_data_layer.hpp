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
      : BaseDataLayer<Dtype>(param), has_new_data_(false) {}
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "NoiseData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  virtual void AddDatumVector(const vector<Datum>& datum_vector);
#ifdef USE_OPENCV
  virtual void AddMatVector(const vector<cv::Mat>& mat_vector);
#endif  // USE_OPENCV

  // Reset should accept const pointers, but can't, because the memory
  //  will be given to Blob, which is mutable
  void Reset(Dtype* data, int n);
  void set_batch_size(int new_size);

  int batch_size() { return batch_size_; }
  int channels() { return channels_; }
  int spatial_size() { return spatial_size_; }
  const char* distribution() { return distribution_; }
  double mu() { return mu_; }
  double sigma() { return sigma_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  int batch_size_, channels_, spatial_size_, size_;
  const char* distribution_;
  double mu_, sigma_;
  Dtype* data_;
  //Dtype* labels_;
  int n_;
  size_t pos_;
  Blob<Dtype> added_data_;
  //Blob<Dtype> added_label_;
  bool has_new_data_;
};

}  // namespace caffe

#endif  // CAFFE_NOISE_DATA_LAYER_HPP_
