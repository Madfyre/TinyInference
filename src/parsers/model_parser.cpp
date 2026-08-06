#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <iostream>
#include <charconv>

#include "mapped_file.cpp"
#include "tokenizer_parser.cpp"

namespace parser {

struct Tensor {
    std::string dtype;

    size_t rows;
    size_t cols;

    size_t offset;
    size_t tensor_size;

    const unsigned char* weights;
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
    MappedFile mapping;
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

// {"dtype":"BF16","shape":[128256,2048],"data_offsets":[0,525336576]}
Tensor TensorParser(JsonLexer& lexer, const char* mmap_base, uint64_t metadata_length) {
    Tensor tensor;
    size_t end_offset = 0;

    bool has_dtype = false;
    bool has_shape = false;
    bool has_offsets = false;

    lexer.Expect(JsonKind::kObjectBegin);
    while (lexer.NextMember()) {
        if (lexer.key() == "dtype") {
            lexer.Expect(JsonKind::kString);
            tensor.dtype = std::string(lexer.text());
            has_dtype = true;
        } else if (lexer.key() == "shape") {
            size_t dims[2] = {1, 1};
            size_t rank = 0;

            lexer.Expect(JsonKind::kArrayBegin);
            while (lexer.NextElement()) {
                lexer.Expect(JsonKind::kNumber);
                if (rank < 2) {
                    dims[rank] = lexer.Number<size_t>();
                }
                ++rank;
            }
            if (rank == 0 || rank > 2) {
                throw std::runtime_error("safetensors: unsupported tensor rank " + std::to_string(rank));
            }

            tensor.rows = dims[0];
            tensor.cols = (rank == 2) ? dims[1] : 1;
            has_shape = true;
        } else if (lexer.key() == "data_offsets") {
            lexer.Expect(JsonKind::kArrayBegin);
            lexer.Expect(JsonKind::kNumber);
            tensor.offset = lexer.Number<size_t>();
            lexer.Expect(JsonKind::kComma);
            lexer.Expect(JsonKind::kNumber);
            end_offset = lexer.Number<size_t>();
            lexer.Expect(JsonKind::kArrayEnd);
            has_offsets = true;
        } else {
            lexer.SkipValue();
        }
    }

    if (!has_dtype || !has_shape || !has_offsets) {
        throw std::runtime_error("safetensors: tensor entry is missing dtype, shape or data_offsets");
    }

    tensor.tensor_size = end_offset - tensor.offset;
    tensor.weights = reinterpret_cast<const unsigned char*>(mmap_base) + 8 + metadata_length + tensor.offset;

    return tensor;
}

Tensor* LayerSlot(Layer& layer, std::string_view field) {
    if (field == "self_attn.q_proj.weight") return &layer.q_proj;
    if (field == "self_attn.k_proj.weight") return &layer.k_proj;
    if (field == "self_attn.v_proj.weight") return &layer.v_proj;
    if (field == "self_attn.o_proj.weight") return &layer.o_proj;
    if (field == "mlp.gate_proj.weight") return &layer.gate_proj;
    if (field == "mlp.up_proj.weight") return &layer.up_proj;
    if (field == "mlp.down_proj.weight") return &layer.down_proj;
    if (field == "input_layernorm.weight") return &layer.input_layernorm;
    if (field == "post_attention_layernorm.weight") return &layer.post_attn_layernorm;
    return nullptr;
}

Model ParseWeights(MappedFile file) {
    if (file.size() < 8) {
        throw std::runtime_error("safetensors: file is too small to hold a header length");
    }

    const char* ptr = file.data();
    uint64_t metadata_length;
    std::memcpy(&metadata_length, ptr, 8);
    if (metadata_length > file.size() - 8) {
        throw std::runtime_error("safetensors: header length " + std::to_string(metadata_length) +
                                 " exceeds the file");
    }

    Model model;
    model.layers.resize(1);

    constexpr std::string_view kLayerPrefix = "model.layers.";

    JsonLexer lexer{std::string_view{ptr + 8, metadata_length}};
    lexer.Expect(JsonKind::kObjectBegin);
    while (lexer.NextMember()) {
        std::string_view name = lexer.key();

        if (name == "model.embed_tokens.weight") {
            model.embed_tensor = TensorParser(lexer, ptr, metadata_length);
            continue;
        }
        if (name == "model.norm.weight") {
            model.final_norm = TensorParser(lexer, ptr, metadata_length);
            continue;
        }
        if (!name.starts_with(kLayerPrefix)) {
            lexer.SkipValue();  // __metadata__ and anything else we ignore
            continue;
        }

        std::string_view rest = name.substr(kLayerPrefix.size());
        size_t dot = rest.find('.');
        if (dot == std::string_view::npos) {
            throw std::runtime_error("safetensors: malformed layer name '" + std::string(name) + "'");
        }

        size_t layer_number = 0;
        auto [end, ec] = std::from_chars(rest.data(), rest.data() + dot, layer_number);
        if (ec != std::errc{}) {
            throw std::runtime_error("safetensors: malformed layer index in '" + std::string(name) + "'");
        }
        if (model.layers.size() <= layer_number) {
            model.layers.resize(layer_number + 1);
        }

        if (Tensor* slot = LayerSlot(model.layers[layer_number], rest.substr(dot + 1)); slot != nullptr) {
            *slot = TensorParser(lexer, ptr, metadata_length);
        } else {
            lexer.SkipValue();
        }
    }

    model.mapping = std::move(file);
    return model;
}

}