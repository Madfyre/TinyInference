#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <iostream>
#include <charconv>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../parsers/tokenizer_parser.cpp"

namespace parser {

struct Tensor {
    std::string dtype;

    size_t rows;
    size_t cols;

    size_t offset;
    size_t tensor_size;

    unsigned char* weights;
};

struct Layer {
    Tensor input_layernorm;
    Tensor post_attn_layernorm;

    Tensor q_proj;
    Tensor k_proj;
    Tensor v_proj;
    Tensor o_proj;

    Tensor gate_proj;
    Tensor up_proj;
    Tensor down_proj;
};

using Embeddings = Tensor;

struct Model {
    Embeddings embed_tensor;
    std::vector<Layer> layers;
    Tensor final_norm;
};

float Bf16ToF32(uint16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &bits, 4);
    return result;
}

void DebugTensor(const Tensor& tensor, const std::string& name, size_t n = 8) {
    std::cout << name << ": dtype=" << tensor.dtype
              << " shape=[" << tensor.rows << ", " << tensor.cols << "]"
              << " offset=" << tensor.offset
              << " size=" << tensor.tensor_size << "\n";

    const uint16_t* data = reinterpret_cast<const uint16_t*>(tensor.weights);

    size_t total = tensor.rows * tensor.cols;
    size_t count = std::min(n, total);

    std::cout << "  first " << count << " values: ";
    for (size_t i = 0; i < count; ++i) {
        std::cout << Bf16ToF32(data[i]) << " ";
    }
    std::cout << "\n";
}

Tensor TensorParser(std::string_view str, char* mmap_base, uint64_t metadata_length) {
    Tensor tensor;

    size_t dtype_start = str.find("dtype");  
    size_t shape_start = str.find("shape", dtype_start);
    size_t data_offsets_start = str.find("data_offsets", shape_start);

    tensor.dtype = str.substr(dtype_start + 8, ((shape_start - 3) - (dtype_start + 8)));

    auto shape_str = str.substr(shape_start + 8, ((data_offsets_start - 3) - (shape_start + 8)));
    size_t comma = shape_str.find(',');
    std::from_chars(shape_str.data(), shape_str.data() + comma, tensor.rows);

    if (shape_str.find(',') == std::string::npos) {
        tensor.cols = 1;
    } else {
        std::from_chars(shape_str.data() + comma + 1, shape_str.data() + shape_str.size(), tensor.cols);
    }
    std::from_chars(str.data() + data_offsets_start + 15, str.data() + str.find(',', data_offsets_start + 15), tensor.offset);

    size_t end_offset;
    std::from_chars(str.data() + str.find(',', data_offsets_start + 15) + 1, str.data() + str.size(), end_offset);

    tensor.tensor_size = end_offset - tensor.offset;
    tensor.weights = reinterpret_cast<unsigned char*>(mmap_base) + 8 + metadata_length + tensor.offset;

    return tensor;
}

Model ParseConfig(const FileInfo& file_info) {
    char* ptr = static_cast<char*>(mmap(nullptr, file_info.file_size, PROT_READ, MAP_PRIVATE, file_info.fd, 0));

    uint64_t metadata_length;
    std::memcpy(&metadata_length, ptr, 8);

    std::string_view config{ptr + 8, metadata_length};
    std::vector<std::string> types;

    size_t left = 1;
    size_t right = 1;
    while (left != std::string::npos) {
        right = config.find(',', config.find('}', left));

        if (right == std::string::npos) {
            types.emplace_back(config.data() + left, config.data() + metadata_length - 1);
            break;
        }

        types.emplace_back(config.data() + left, config.data() + right);
        left = right + 1;
    }

    Model model;
    model.layers.resize(1);
    for (const auto& str : types) {
        if (auto it = str.find("model.embed_tokens.weight"); it != std::string::npos) {
            model.embed_tensor = TensorParser(str, ptr, metadata_length);
            continue;
        }

        if (str.find("model.norm.weight") != std::string::npos) {
            model.final_norm = TensorParser(str, ptr, metadata_length);
            continue;
        }

        if (auto it = str.find("model.layers"); it != std::string::npos) {
            size_t layer_number =  std::stoll(str.substr(it + 13, str.find('.', it + 13) - (it + 13)));
            if (model.layers.size() <= layer_number) {
                model.layers.resize(layer_number + 1);
            }
            if (str.find("q_proj") != std::string::npos) {
                model.layers[layer_number].q_proj = TensorParser(str, ptr, metadata_length);
            } else if (str.find("k_proj") != std::string::npos) {
                model.layers[layer_number].k_proj = TensorParser(str, ptr, metadata_length);
            } else if (str.find("v_proj") != std::string::npos) {
                model.layers[layer_number].v_proj = TensorParser(str, ptr, metadata_length);
            } else if (str.find("o_proj") != std::string::npos) {
                model.layers[layer_number].o_proj = TensorParser(str, ptr, metadata_length);
            } else if (str.find("gate_proj") != std::string::npos) {
                model.layers[layer_number].gate_proj = TensorParser(str, ptr, metadata_length);
            } else if (str.find("up_proj") != std::string::npos) {
                model.layers[layer_number].up_proj = TensorParser(str, ptr, metadata_length);
            } else if (str.find("down_proj") != std::string::npos) {
                model.layers[layer_number].down_proj = TensorParser(str, ptr, metadata_length);
            } else if (str.find("input_layernorm") != std::string::npos) {
                model.layers[layer_number].input_layernorm = TensorParser(str, ptr, metadata_length);
            } else if (str.find("post_attention_layernorm") != std::string::npos) {
                model.layers[layer_number].post_attn_layernorm = TensorParser(str, ptr, metadata_length);
            }
        }
    }
    return model;
}

}