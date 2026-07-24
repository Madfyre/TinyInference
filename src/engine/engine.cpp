#include <cstddef>
#include <vector>

namespace engine {

struct Tensor {
    std::string dtype;
    size_t rows;
    size_t cols;
    char* data;
};

struct Layer {
    Tensor Q;
    Tensor K;
    Tensor V;
    Tensor O;
    Tensor Lin;
};

struct Model {
    Tensor embeddings;
    std::vector<Layer> layers;
}

}