/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	optimizer.cpp
 * @date	08 April 2020
 * @brief	This is Implementation of Optimizer class
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <optimizer.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

Optimizer::Optimizer() {
  type = OptType::unknown;
  popt.learning_rate = 0.0;
  popt.beta1 = 0.0;
  popt.beta2 = 0.0;
  popt.epsilon = 0.0;
  popt.decay_rate = 0.0;
  popt.decay_steps = 0.0;
};

int Optimizer::setType(OptType t) {
  int status = ML_ERROR_NONE;
  if (t == OptType::unknown) {
    ml_loge("Error: Optimizer is unknown");
    return ML_ERROR_INVALID_PARAMETER;
  }
  type = t;
  return status;
}

int Optimizer::setOptParam(OptParam p) {
  int status = ML_ERROR_NONE;
  if (p.learning_rate <= 0) {
    ml_loge("Error: learning_rate should be grater than 0 (%f)",
            p.learning_rate);
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (p.decay_steps == -1 && p.beta1 && p.beta2 && p.epsilon) {
    ml_logw("Although you set the learning rate decay param, you didn't "
            "set decay_steps");
  }

  popt = p;
  return status;
}

int Optimizer::initialize(TensorDim d, bool set_tensor) {
  int status = ML_ERROR_NONE;
  if (d.height() == 0 || d.width() == 0 || d.channel() == 0) {
    ml_loge("Error: Tensor Dimension must be greater than 0");
    return ML_ERROR_INVALID_PARAMETER;
  }
  if (type == OptType::adam && set_tensor) {
    wm = Tensor(d.channel(), d.height(), d.width());
    wv = Tensor(d.channel(), d.height(), d.width());
    wm.setZero();
    wv.setZero();
    bm = Tensor(1, 1, d.width());
    bv = Tensor(1, 1, d.width());
    bm.setZero();
    bv.setZero();
  }
  return status;
}

void Optimizer::calculate(Tensor &djdw, Tensor &djdb, Tensor &weight,
                          Tensor &bias, int iteration, bool init_zero,
                          WeightDecayParam weight_decay) {
  Tensor djdwAvg, djdbAvg;
  if (weight_decay.type == WeightDecayType::l2norm) {
    djdw = djdw.add(weight.multiply(weight_decay.lambda));
  }

  float ll = popt.learning_rate;
  if (popt.decay_steps != -1) {
    ll = ll * pow(popt.decay_rate, (iteration / popt.decay_steps));
  }

  djdwAvg = djdw.average();
  djdbAvg = djdb.average();

  switch (type) {
  case OptType::sgd:
    weight = weight.subtract(djdwAvg.multiply(ll));
    break;
  case OptType::adam:
    wm = wm.multiply(popt.beta1).add(djdwAvg.multiply(1 - popt.beta1));
    wv = wv.multiply(popt.beta2)
           .add((djdwAvg.multiply(djdwAvg)).multiply(1 - popt.beta2));
    wm.divide(1 - pow(popt.beta1, iteration + 1));
    wv.divide(1 - pow(popt.beta2, iteration + 1));
    weight = weight.subtract(
      (wm.divide(wv.apply(sqrtFloat).add(popt.epsilon))).multiply(ll));
    bm = bm.multiply(popt.beta1).add(djdbAvg.multiply(1 - popt.beta1));
    bv = bv.multiply(popt.beta2)
           .add((djdbAvg.multiply(djdbAvg)).multiply(1 - popt.beta2));
    bm.divide(1 - pow(popt.beta1, iteration + 1));
    bv.divide(1 - pow(popt.beta2, iteration + 1));
    bias = bias.subtract(
      (bm.divide(bv.apply(sqrtFloat).add(popt.epsilon))).multiply(ll));
    break;
  default:
    break;
  }

  if (init_zero) {
    bias = bias.subtract(djdbAvg.multiply(ll));
  }
}

int Optimizer::setProperty(std::vector<std::string> values) {
  int status = ML_ERROR_NONE;

  for (unsigned int i = 0; i < values.size(); ++i) {
    std::string key;
    std::string value;
    status = getKeyValue(values[i], key, value);

    unsigned int type = parseOptProperty(key.c_str());

    switch (static_cast<PropertyType>(type)) {
    case PropertyType::learning_rate:
      status = setFloat(popt.learning_rate, value);
      NN_RETURN_STATUS();
      break;
    case PropertyType::decay_steps:
      status = setFloat(popt.decay_steps, value);
      NN_RETURN_STATUS();
      break;
    case PropertyType::decay_rate:
      status = setFloat(popt.decay_rate, value);
      NN_RETURN_STATUS();
      break;
    case PropertyType::beta1:
      status = setDouble(popt.beta1, value);
      NN_RETURN_STATUS();
      break;
    case PropertyType::beta2:
      status = setDouble(popt.beta2, value);
      NN_RETURN_STATUS();
      break;
    case PropertyType::epsilon:
      status = setDouble(popt.epsilon, value);
      NN_RETURN_STATUS();
      break;
    default:
      ml_loge("Error: Unknown Optimizer Property Key");
      status = ML_ERROR_INVALID_PARAMETER;
      break;
    }
  }

  return status;
}
} // namespace nntrainer
