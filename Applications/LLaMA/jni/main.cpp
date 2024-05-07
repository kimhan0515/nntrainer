// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   main.cpp
 * @date   7 August 2023
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <ctime>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <layer.h>
#include <model.h>
#include <optimizer.h>

#include <app_context.h>
#include <rms_norm.h>
#include <rotary_embedding.h>
#include <swiglu.h>
#include <transpose_layer.h>

#if defined(ENABLE_ENCODER2)
#include "json.hpp"
#include <codecvt>
#include <encoder.hpp>
#include <locale>
using json = nlohmann::json;
#endif

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

ModelHandle g_model;

// Hyper params for LLaMA
int const DIM = 2304;
int const NUM_LAYERS = 28;
int const NUM_HEADS = 18;

int const MULTIPLE_OF = 256;

float const NORM_EPS = 0.000001;
float const EPS = 0.000001;
int const NUM_VOCAB = 96000;
int MAX_SEQ_LEN = 1024;
int NUM_TO_GENERATE = 100;

constexpr unsigned int INIT_SEQ_LEN = 28;
unsigned int batch_size = 1;
unsigned int epoch = 1;

/** cache loss values post training for test */
float training_loss = 0.0;
float validation_loss = 0.0;
bool swap = false;

bool optimize = false;

/**
 * @brief make "key=value" from key and value
 *
 * @tparam T type of a value
 * @param key key
 * @param value value
 * @return std::string with "key=value"
 */
template <typename T>
static std::string withKey(const std::string &key, const T &value) {
  std::stringstream ss;
  ss << key << "=" << value;
  return ss.str();
}

template <typename T>
static std::string withKey(const std::string &key,
                           std::initializer_list<T> value) {
  if (std::empty(value)) {
    throw std::invalid_argument("empty data cannot be converted");
  }

  std::stringstream ss;
  ss << key << "=";

  auto iter = value.begin();
  for (; iter != value.end() - 1; ++iter) {
    ss << *iter << ',';
  }
  ss << *iter;

  return ss.str();
}

/**
 * @brief Apply temperature & top-k & top-p to logits
 * @return Max logit for softmax
 */
float applyTKP(float *logits, int len, float temperature, unsigned int top_k,
               float top_p) {
  // Apply temperature & Sort logits
  std::vector<std::pair<int, float>> top_indices_and_logits;
  for (int i = 0; i < len; ++i) {
    if (temperature > 1e-5)
      logits[i] = logits[i] / temperature;
    top_indices_and_logits.push_back({i, logits[i]});
  }
  sort(top_indices_and_logits.begin(), top_indices_and_logits.end(),
       [](auto &a, auto &b) { return a.second > b.second; });

  // Accumulate logits
  float cum_prob = 0;
  unsigned int top_index = 0;
  while (cum_prob <= top_p && top_index < top_k) {
    cum_prob += top_indices_and_logits[top_index].second;
    ++top_index;
  }

  // Apply Top-K and Top-P
  std::fill_n(logits, sizeof(len), -INFINITY);
  for (unsigned int i = 0; i < top_index; ++i) {
    logits[top_indices_and_logits[i].first] = top_indices_and_logits[i].second;
  }

  return top_indices_and_logits[0].second;
}

/**
 * @brief Apply repetition penalty to logits
 */
void applyRepetitionPenalty(float *logits, unsigned int *input_ids,
                            unsigned int NUM_INPUT_IDS,
                            float repetition_penalty = 1) {
  for (unsigned int i = 0; i < NUM_INPUT_IDS; ++i) {
    if (logits[input_ids[i]] < 0) {
      logits[input_ids[i]] *= repetition_penalty;
    } else {
      logits[input_ids[i]] /= repetition_penalty;
    }
  }
}

/**
 * @brief Apply bad words penalty
 */
void applyBadWordsPenalty(float *logits, unsigned int *bad_words_ids,
                          unsigned int NUM_BAD_WORDS_IDS) {
  for (unsigned int i = 0; i < NUM_BAD_WORDS_IDS; ++i) {
    logits[bad_words_ids[i]] = -INFINITY;
  }
}

std::vector<int> generate(float *logits, unsigned int NUM_VOCAB = 0,
                          unsigned int NUM_BATCH = 1, bool do_sample = false,
                          float temperature = 1, unsigned int top_k = 1,
                          float top_p = 1, float repetition_penalty = 1,
                          unsigned int *input_ids = nullptr,
                          unsigned int NUM_INPUT_IDS = 0,
                          unsigned int *bad_words_ids = nullptr,
                          unsigned int NUM_BAD_WORDS_IDS = 0) {
  std::vector<int> outputs;
  for (unsigned int iteration = 0; iteration < NUM_BATCH; ++iteration) {

    // apply repetition penalty
    if (repetition_penalty != 1 && input_ids != nullptr && NUM_INPUT_IDS != 0) {
      applyRepetitionPenalty(logits, input_ids, NUM_INPUT_IDS,
                             repetition_penalty);
    }

    // apply bad words penalty
    if (bad_words_ids != nullptr && NUM_BAD_WORDS_IDS != 0) {
      applyBadWordsPenalty(logits, bad_words_ids, NUM_BAD_WORDS_IDS);
    }

    // return argmax if do_sample is false
    if (do_sample == false) {
      int argmax_idx =
        std::distance(logits, std::max_element(logits, logits + NUM_VOCAB));
      outputs.push_back(argmax_idx);
    } else {
      // apply temperature & top-k & top-p to logits
      float max_logits = applyTKP(logits, NUM_VOCAB, temperature, top_k, top_p);
      // transform logits to softmax
      float sum_exp_logits = 0;
      for (unsigned int i = 0; i < NUM_VOCAB; i++) {
        float exp_x = exp(logits[i] - max_logits);
        sum_exp_logits += exp_x;
        logits[i] = exp_x;
      }

      for (unsigned int i = 0; i < NUM_VOCAB; ++i) {
        logits[i] /= sum_exp_logits;
      }

      // sample from final logits
      std::random_device rd;
      std::mt19937 gen(rd());
      std::discrete_distribution<int> dist(logits, logits + NUM_VOCAB);
      int sampled_idx = dist(gen);

      // add sampled word
      outputs.push_back(sampled_idx);
    }

    // set batch offset
    logits = logits + NUM_VOCAB;
    input_ids = input_ids + batch_size;
  }

  return outputs;
}

std::vector<int> generate_multi_tokens(
  float *logits, unsigned int NUM_VOCAB = 0, unsigned int NUM_TARGET_TOKENS = 1,
  float repetition_penalty = 1, unsigned int *input_ids = nullptr,
  unsigned int NUM_INPUT_IDS = 0, unsigned int *bad_words_ids = nullptr,
  unsigned int NUM_BAD_WORDS_IDS = 0) {
  std::vector<int> outputs;
  // apply repetition penalty
  if (abs(repetition_penalty - 1) < EPS && input_ids != nullptr &&
      NUM_INPUT_IDS != 0) {
    applyRepetitionPenalty(logits, input_ids, NUM_INPUT_IDS,
                           repetition_penalty);
  }

  // apply bad words penalty
  if (bad_words_ids != nullptr && NUM_BAD_WORDS_IDS != 0) {
    applyBadWordsPenalty(logits, bad_words_ids, NUM_BAD_WORDS_IDS);
  }

  // Sort and generate multiple tokens
  std::vector<std::pair<int, float>> top_indices_and_logits;
  for (unsigned int i = 0; i < NUM_VOCAB; ++i) {
    top_indices_and_logits.push_back({i, logits[i]});
  }
  sort(top_indices_and_logits.begin(), top_indices_and_logits.end(),
       [](auto &a, auto &b) { return a.second > b.second; });

  // add sampled words
  for (unsigned int i = 0; i < NUM_TARGET_TOKENS; ++i) {
    outputs.push_back(top_indices_and_logits[i].first);
  }

  return outputs;
}

template <typename T>
T unwrap(std::optional<T> &&value, const std::string &error_msg) {
  if (value.has_value()) {
    return value.value();
  } else {
    throw std::runtime_error(error_msg);
  }
}

/**
 * @brief Create Attention Layer for the seperate impelemntation
 */
std::vector<LayerHandle> createAttentionLayer(const int layer_id, int seq_len,
                                              int n_heads, int head_dim,
                                              std::string query_name,
                                              std::string key_name,
                                              std::string value_name) {
  using ml::train::createLayer;

  std::vector<LayerHandle> layers;

  if (optimize) {
    // linear transformation of q
    for (int i = 0; i < n_heads; i++) {
      layers.push_back(
        createLayer("fully_connected",
                    {withKey("name", "layer" + std::to_string(layer_id) +
                                       "_wq_" + std::to_string(i)),
                     withKey("unit", head_dim), withKey("disable_bias", "true"),
                     withKey("input_layers", query_name)}));
    }

    // linear transformation of k
    for (int i = 0; i < n_heads; i++) {
      layers.push_back(
        createLayer("fully_connected",
                    {withKey("name", "layer" + std::to_string(layer_id) +
                                       "_wk_" + std::to_string(i)),
                     withKey("unit", head_dim), withKey("disable_bias", "true"),
                     withKey("input_layers", key_name)}));
    }

    // linear transformation of v
    for (int i = 0; i < n_heads; i++) {
      layers.push_back(
        createLayer("fully_connected",
                    {withKey("name", "layer" + std::to_string(layer_id) +
                                       "_wv_" + std::to_string(i)),
                     withKey("unit", head_dim), withKey("disable_bias", "true"),
                     withKey("input_layers", value_name)}));
    }

    std::string concat_input = "";
    // apply rotary embedding and dot_product attention
    for (int i = 0; i < n_heads; i++) {
      // reshape q, k, v (apply num_heads)
      layers.push_back(createLayer(
        "reshape", {withKey("name", "layer" + std::to_string(layer_id) +
                                      "_q_reshape_" + std::to_string(i)),
                    withKey("target_shape", "1:" + std::to_string(seq_len) +
                                              ":" + std::to_string(head_dim)),
                    withKey("input_layers", "layer" + std::to_string(layer_id) +
                                              "_wq_" + std::to_string(i))}));

      layers.push_back(createLayer(
        "reshape", {withKey("name", "layer" + std::to_string(layer_id) +
                                      "_k_reshape_" + std::to_string(i)),
                    withKey("target_shape", "1:" + std::to_string(seq_len) +
                                              ":" + std::to_string(head_dim)),
                    withKey("input_layers", "layer" + std::to_string(layer_id) +
                                              "_wk_" + std::to_string(i))}));

      layers.push_back(createLayer(
        "reshape", {withKey("name", "layer" + std::to_string(layer_id) +
                                      "_v_reshape_" + std::to_string(i)),
                    withKey("target_shape", "1:" + std::to_string(seq_len) +
                                              ":" + std::to_string(head_dim)),
                    withKey("input_layers", "layer" + std::to_string(layer_id) +
                                              "_wv_" + std::to_string(i))}));

      // apply rotary embedding to q, k
      layers.push_back(createLayer(
        "rotary_embedding",
        {withKey("name", "layer" + std::to_string(layer_id) + "_q_rotary_" +
                           std::to_string(i)),
         withKey("input_layers", "layer" + std::to_string(layer_id) +
                                   "_q_reshape_" + std::to_string(i))}));

      layers.push_back(createLayer(
        "rotary_embedding",
        {withKey("name", "layer" + std::to_string(layer_id) + "_k_rotary_" +
                           std::to_string(i)),
         withKey("input_layers", "layer" + std::to_string(layer_id) +
                                   "_k_reshape_" + std::to_string(i))}));

      // apply scaled-dot product attention
      layers.push_back(ml::train::layer::Attention(
        {"name=layer" + std::to_string(layer_id) + "_attention_" +
           std::to_string(i),
         "input_layers=layer" + std::to_string(layer_id) + "_q_rotary_" +
           std::to_string(i) + ",layer" + std::to_string(layer_id) +
           "_v_reshape_" + std::to_string(i) + ",layer" +
           std::to_string(layer_id) + "_k_rotary_" + std::to_string(i),
         "scaled_dot_product=true", "causal_mask=true"}));

      layers.push_back(createLayer(
        "reshape",
        {withKey("name", "layer" + std::to_string(layer_id) +
                           "_attention_output_" + std::to_string(i)),
         withKey("target_shape", "1:" + std::to_string(seq_len) + ":" +
                                   std::to_string(head_dim)),
         withKey("input_layers", "layer" + std::to_string(layer_id) +
                                   "_attention_" + std::to_string(i))}));

      concat_input += "layer" + std::to_string(layer_id) +
                      "_attention_output_" + std::to_string(i);

      if (i != n_heads - 1) {
        concat_input += ",";
      }
    }

    // concat attention output
    layers.push_back(createLayer(
      "concat", {withKey("name", "layer" + std::to_string(layer_id) +
                                   "_attention_concat"),
                 withKey("axis", 3), withKey("input_layers", concat_input)}));

    // reshape for flatten
    layers.push_back(createLayer(
      "reshape", {withKey("name", "layer" + std::to_string(layer_id) +
                                    "_attention_flatten"),
                  withKey("target_shape", "1:" + std::to_string(seq_len) + ":" +
                                            std::to_string(n_heads * head_dim)),
                  withKey("input_layers", "layer" + std::to_string(layer_id) +
                                            "_attention_concat")}));

    // linear transformation of attention output
    layers.push_back(createLayer(
      "fully_connected",
      {withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
       withKey("unit", head_dim * n_heads), withKey("disable_bias", "true"),
       withKey("input_layers",
               "layer" + std::to_string(layer_id) + "_attention_flatten")}));
  } else {
    layers.push_back(createLayer(
      "multi_head_attention",
      {withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
       withKey("num_heads", std::to_string(NUM_HEADS)),
       withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
       withKey("disable_bias", "true"),
       withKey("input_layers", {query_name, key_name, value_name})}));
  }

  return layers;
}

/**
 * @brief Create FF Layers
 */
std::vector<LayerHandle> createFeedForwardLayer(const int layer_id, int dim,
                                                int hidden_dim,
                                                std::string input_name,
                                                int multiplier = 1) {
  using ml::train::createLayer;
  std::vector<LayerHandle> layers;

  hidden_dim = 2 * multiplier * hidden_dim / 3;
  hidden_dim = MULTIPLE_OF * ((hidden_dim + MULTIPLE_OF - 1) / MULTIPLE_OF);

  layers.push_back(
    createLayer("fully_connected",
                {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_1"),
                 withKey("unit", hidden_dim), withKey("disable_bias", "true"),
                 withKey("input_layers", input_name)}));
  layers.push_back(
    createLayer("fully_connected",
                {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_2"),
                 withKey("unit", hidden_dim), withKey("disable_bias", "true"),
                 withKey("input_layers", input_name)}));

  layers.push_back(createLayer(
    "swiglu",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
     withKey("input_layers", "layer" + std::to_string(layer_id) + "_ffn_1," +
                               "layer" + std::to_string(layer_id) +
                               "_ffn_2")}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_output"),
     withKey("unit", dim), withKey("disable_bias", "true"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_ffn_swiglu")}));

  return layers;
}

/**
 * @brief Create Decoder
 */
std::vector<LayerHandle> createTransformerDecoder(const int layer_id,
                                                  std::string input_name) {
  using ml::train::createLayer;
  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("input_layers", input_name),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  auto att_layer = createAttentionLayer(
    layer_id, INIT_SEQ_LEN, NUM_HEADS, DIM / NUM_HEADS,
    "layer" + std::to_string(layer_id) + "_attention_norm",
    "layer" + std::to_string(layer_id) + "_attention_norm",
    "layer" + std::to_string(layer_id) + "_attention_norm");
  layers.insert(layers.end(), att_layer.begin(), att_layer.end());

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_add"),
     withKey("input_layers", input_name + ",layer" + std::to_string(layer_id) +
                               "_attention_out")}));

  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_decoder_add"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("packed", "false")}));

  auto ffn_layer = createFeedForwardLayer(
    layer_id, DIM, 4 * DIM, "layer" + std::to_string(layer_id) + "_ffn_norm");
  layers.insert(layers.end(), ffn_layer.begin(), ffn_layer.end());

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output"),
     withKey("input_layers", "layer" + std::to_string(layer_id) +
                               "_decoder_add,layer" + std::to_string(layer_id) +
                               "_ffn_output")}));

  return layers;
}

/**
 * @brief Create LLaMA2 Model
 */
ModelHandle createLLaMA() {
  using ml::train::createLayer;

  ModelHandle model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "input", {withKey("name", "input0"),
              withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));

  layers.push_back(ml::train::layer::Embedding(
    {"name=embedding0", "in_dim=" + std::to_string(NUM_VOCAB), "packed=false",
     "out_dim=" + std::to_string(DIM)}));

  for (int i = 0; i < NUM_LAYERS; i++) {
    std::vector<LayerHandle> transformer;
    if (i == 0)
      transformer = createTransformerDecoder(i, "embedding0");
    else
      transformer = createTransformerDecoder(
        i, "layer" + std::to_string(i - 1) + "_decoder_output");
    layers.insert(layers.end(), transformer.begin(), transformer.end());
  }

  int last_layer = NUM_LAYERS - 1;

  layers.push_back(createLayer(
    "rms_norm", {withKey("name", "output_norm"),
                 withKey("epsilon", std::to_string(NORM_EPS)),
                 withKey("input_layers", "layer" + std::to_string(last_layer) +
                                           "_decoder_output"),
                 withKey("packed", "false")}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "output_of_llama"), withKey("unit", NUM_VOCAB),
     withKey("disable_bias", "true"), withKey("input_layers", "output_norm"),
     withKey("packed", "false")}));

  for (auto &layer : layers) {
    model->addLayer(layer);
  }

  return model;
}

/**
 * @brief to run for every text sequence
 */
void run(std::string text, bool apply_temperature) {

  std::vector<float *> input;
  std::vector<float *> label;

  int data_size = batch_size * INIT_SEQ_LEN;

  float *input_sample = (float *)malloc(sizeof(float) * data_size);

  if (input_sample == nullptr) {
    std::cerr << "failed to malloc()" << std::endl;
    exit(EXIT_SUCCESS);
  }

  unsigned int input_len = INIT_SEQ_LEN;

  unsigned int init_len;

#if defined(ENABLE_ENCODER2)
  std::string vocab_file_name = "../Applications/LLaMA/jni/vocab.json";
  std::string merge_file_name = "../Applications/LLaMA/jni/merges.txt";

  auto tokenizer = unwrap(GPT2Encoder::load(vocab_file_name, merge_file_name),
                          "Error initializising GPT2 tokenizer\n");

  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;

  auto init_input = tokenizer.encode(converter.from_bytes(text));
  init_len = init_input.size();

  input_len = (init_len > INIT_SEQ_LEN) ? INIT_SEQ_LEN : init_len;

  for (unsigned int i = 0; i < input_len; ++i) {
    input_sample[i] = static_cast<float>(init_input[i]);
  }

  input.push_back(input_sample);

#else
  float init_input[INIT_SEQ_LEN] = {
    0,  1,  2,  3,  4,  5,   6,   7,   8,   9,   10,  20,  30,  40,
    50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900};
  memcpy(input_sample, init_input, sizeof(float) * INIT_SEQ_LEN);
  input.push_back(input_sample);
  init_len = 18;
#endif

  std::vector<int64_t> token_ids;

  auto output =
    g_model->incremental_inference(1, input, label, MAX_SEQ_LEN, 0, input_len);

  unsigned int ids = std::distance(
    output[0], std::max_element(output[0], output[0] + NUM_VOCAB));

  input_sample[0] = static_cast<float>(ids);

#ifdef ENABLE_FP16
  for (auto o : output) {
    delete[] o;
  }
#endif
  std::cout << " Progress Reading: 100 % " << std::endl;
  std::cout << std::endl << "### Output : " << std::endl;
  if (init_len < INIT_SEQ_LEN) {
#if defined(ENABLE_ENCODER2)
    auto decoded_str = tokenizer.decode({static_cast<int64_t>(ids)});
    std::cout << decoded_str << " ";
    std::cout.flush();
#endif
  }

  for (unsigned int i = input_len + 1; i < input_len + NUM_TO_GENERATE; ++i) {
    auto output_interval =
      g_model->incremental_inference(1, input, label, MAX_SEQ_LEN, i - 1, i);
    unsigned int ids = generate(output[0], NUM_VOCAB)[0];

    if (i < input_len) {
      input_sample[0] = static_cast<float>(init_input[i]);
    } else {
      input_sample[0] = static_cast<float>(ids);
#if defined(ENABLE_ENCODER2)
      auto decoded_str = tokenizer.decode({static_cast<int64_t>(ids)});
      std::cout << decoded_str << " ";
      std::cout.flush();
#endif
    }

#ifdef ENABLE_FP16
    for (auto o : output_interval) {
      delete[] o;
    }
#endif
  }

  std::cout << std::endl;
  free(input_sample);
}

/**
 * @brief to creaet model
 */
void createAndRun(unsigned int epochs, unsigned int batch_size) {
  // setup model
  g_model = createLLaMA();
  g_model->setProperty({withKey("batch_size", batch_size),
                        withKey("epochs", epochs),
                        // #ifdef ENABLE_FP16
                        withKey("model_tensor_type", "FP16-FP16"),
                        // #endif
                        withKey("save_path", "test_model.bin")});

  auto optimizer = ml::train::createOptimizer("sgd", {"learning_rate=0.001"});
  g_model->setOptimizer(std::move(optimizer));

  int status = g_model->compile();
  if (status) {
    throw std::invalid_argument("model compilation failed!");
  }

  status = g_model->initialize();
  if (status) {
    throw std::invalid_argument("model initialization failed!");
  }

  std::string weight_path = "./llama_fp16.bin";

  g_model->load(weight_path);
}

#if defined(ENABLE_ENCODER2)
std::wstring decodeUnicodeEscape(const std::wstring &input) {
  std::wstringstream result;

  for (size_t i = 0; i < input.length(); ++i) {
    if (i + 5 < input.length() && input[i] == L'\\' && input[i + 1] == L'u') {
      std::wstring unicodeSeq;
      for (int j = 0; j < 4; ++j)
        unicodeSeq += input[i + 2 + j];

      result << static_cast<wchar_t>(std::stoi(unicodeSeq, nullptr, 16));
      i += 5;
    } else if (input[i] == L'\\' && input[i + 1] == L'n') {
      result << static_cast<wchar_t>('\n');
      i++;
    } else if (input[i] == L' ')
      result << static_cast<wchar_t>(' ');
    else
      result << input[i];
  }

  return result.str();
}
#endif
int main(int argc, char *argv[]) {
  // Setting locale
  std::locale::global(std::locale("ko_KR.UTF-8"));

#if defined(ENABLE_ENCODER2)
  // Getting arguments From terminal
  std::wstring input;
  std::getline(std::wcin, input);
  std::wstring test = decodeUnicodeEscape(input);
  std::wstring_convert<std::codecvt_utf16<wchar_t>> converter;
  std::string text = converter.to_bytes(test);
#else
  std::string text = "This is smaple input for LLaMA.";
#endif

  auto &app_context = nntrainer::AppContext::Global();
  try {
    app_context.registerFactory(nntrainer::createLayer<custom::SwiGLULayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
    return 1;
  }

  try {
    app_context.registerFactory(nntrainer::createLayer<custom::RMSNormLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
    return 1;
  }

  try {
    const std::vector<std::string> args(argv + 1, argv + argc);

    bool apply_temp = (strcasecmp("true", args[1].c_str()) == 0);

    createAndRun(epoch, batch_size);

    run(text, apply_temp);

  } catch (const std::exception &e) {
    std::cerr << "uncaught error while running! details: " << e.what()
              << std::endl;
    return EXIT_FAILURE;
  }

  int status = EXIT_SUCCESS;
  return status;
}
