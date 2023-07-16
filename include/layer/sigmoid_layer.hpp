// created by Enting Zhou on July 16th 2023

#ifndef KUIPER_COURSE_INCLUDE_LAYER_SIGMOID_LAYER_HPP_
#define KUIPER_COURSE_INCLUDE_LAYER_SIGMOID_LAYER_HPP_
#include "layer.hpp"
#include "ops/op.hpp"
#include "ops/sigmoid_op.hpp"

namespace kuiper_infer {
class SigmoidLayer: public Layer {
  public:
  ~SigmoidLayer() = default;

  explicit SigmoidLayer(const std::shared_ptr<Operator> &op);

  void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

  static std::shared_ptr<Layer> CreateInstance(const std::shared_ptr<Operator> &op);

  private:
  std::unique_ptr<SigmoidOperator> op_;
};
}
#endif