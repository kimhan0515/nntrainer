// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 heka1024 <heka1024@gmail.com>
 *
 * @file   transposed_conv2d_layer.h
 * @date   2 June 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author heka1024 <heka1024@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Transposed Convolution 2D Layer Class for Neural Network
 *
 */

#ifndef __TRANSPOSED_CONV2D_LAYER_H_
#define __TRANSPOSED_CONV2D_LAYER_H_
#ifdef __cplusplus

#include <memory.h>

#include <common_properties.h>
#include <layer_impl.h>


namespace nntrainer {

constexpr const unsigned int TRANSPOSED_CONV2D_DIM = 2;

/**
 * @class   Transposed Convolution 2D Layer
 * @brief   Transposed Convolution 2D Layer
 */
class TransposedConv2dLayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of Transposed Convolution 2D Layer
   */
  TransposedConv2dLayer(
    const std::array<unsigned int, TRANSPOSED_CONV2D_DIM * 2> &padding_ = {0, 0, 0, 0},
    const std::array<unsigned int, TRANSPOSED_CONV2D_DIM * 2> &output_padding_ = {0, 0, 0, 0}
 );

  /**
   * @brief     Destructor of Transposed Convolution 2D Layer
   */
  ~TransposedConv2dLayer() = default;

  /**
   *  @brief  Move constructor of Transposed Convolution 2D Layer
   *  @param[in] Conv2dLayer &&
   */
  TransposedConv2dLayer(TransposedConv2dLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs TransposedConv2DLayer to be moved.
   */
  TransposedConv2dLayer &operator=(TransposedConv2dLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return TransposedConv2dLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "transposedconv2d";

private:
    std::array<unsigned int, TRANSPOSED_CONV2D_DIM * 2> padding;
    std::array<unsigned int, TRANSPOSED_CONV2D_DIM * 2> output_padding;

    std::tuple<props::FilterSize,
               std::array<props::KernelSize, TRANSPOSED_CONV2D_DIM>,
               std::array<props::Stride, TRANSPOSED_CONV2D_DIM>,
               props::Padding2D, // input type
               props::Padding2D, // output type
               std::array<props::Dilation, TRANSPOSED_CONV2D_DIM>>
      transposed_conv_2d_props;
};
}

#endif /* __cplusplus */
#endif /* __TRANSPOSED_CONV2D_LAYER_H_ */