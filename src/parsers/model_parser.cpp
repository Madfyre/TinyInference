#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

#include <fcntl.h>
#include <unistd.h>

namespace parser {

struct Tensor {
    std::string dtype;

    size_t rows;
    size_t cols;

    size_t offset;
    size_t tensor_size;

    unsigned char* weights;
};

using Layer = std::vector<Tensor>;
using Embeddings = Tensor;

struct Model {
    Embeddings embed_tensor;
    std::vector<Layer> layers;
};

Tensor TensorParser(const std::string& str, int fd) {
    Tensor tensor;

    size_t dtype_start = str.find("dtype");  
    size_t shape_start = str.find("shape", dtype_start);
    size_t data_offsets_start = str.find("data_offsets", shape_start);

    tensor.dtype = str.substr(dtype_start + 8, ((shape_start - 3) - (dtype_start + 8)));

    std::string shape_str = str.substr(shape_start + 8, ((data_offsets_start - 3) - (shape_start + 8)));
    tensor.rows = std::stoll(shape_str.substr(0, shape_str.find(',')));
    if (shape_str.find(',') == std::string::npos) {
        tensor.cols = 1;
    } else {
        tensor.cols = std::stoll(shape_str.substr(shape_str.find(',') + 1, shape_str.size() - shape_str.find(',') - 1));
    }

    std::string offset_str = str.substr(data_offsets_start + 15, ((str.size() - 2) - (data_offsets_start + 15)));
    tensor.offset = std::stoll(offset_str.substr(0, offset_str.find(',')));
    tensor.tensor_size = std::stoll(offset_str.substr(offset_str.find(',') + 1, offset_str.size() - offset_str.find(',')));

    lseek(fd, tensor.offset, SEEK_SET);
    size_t result = read(fd, tensor.weights, tensor.tensor_size);
    if (!result) {
        throw std::runtime_error("weights in " + offset_str + " was not parsed :(");
    }

    return tensor;
}

Model ParseConfig(const std::string& filepath) {
    int fd = open(filepath.c_str(), O_RDONLY);

    uint64_t metadata_length;
    int result = read(fd, &metadata_length, 8);
    if (result != 8) {
        throw std::runtime_error("Metadata length was not parsed :(");
    }

    std::string config;
    config.resize(metadata_length);
    result = read(fd, config.data(), metadata_length);
    if (result != metadata_length) {
        throw std::runtime_error("Config was not parsed :(");
    }

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
            model.embed_tensor = TensorParser(str, fd);
            continue;
        }

        if (auto it = str.find("model.layers"); it != std::string::npos) {
            size_t layer_number =  std::stoll(str.substr(it + 13, str.find('.', it + 13) - (it + 13)));
            if (model.layers.size() <= layer_number) {
                model.layers.resize(layer_number + 1);
            }
            model.layers[layer_number].push_back(TensorParser(str, fd));
        }
    }

    return model;
}

}