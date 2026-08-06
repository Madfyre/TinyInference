#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

#include "../model/debug.cpp"
#include "../model/llama.cpp"

int main() {
    std::string input = "2+2";
    std::string model_dir = "models/llama-3.2-1b-instruct";
    std::string ref_dir = "ref_instruct";

    model::Llama llama = model::LlamaBuilder{}.Dir(model_dir).MaxSeqLen(512).Build();

    bool trace = true;
    llama.set_layer_hook([&](size_t index, const model::Matrix& hidden) {
        if (!trace) {
            return;
        }
        model::CompareRef(hidden, ref_dir + "/h" + std::to_string(index) + ".bin");
        trace = index != static_cast<size_t>(llama.config().n_layers);
    });

    model::ChatPrompt prompt{llama.tokenizer()};
    prompt.Message("system", "Cutting Knowledge Date: December 2023\nToday Date: 02 Aug 2026\n\n")
          .Message("user", input)
          .AssistantHeader();

    llama.Generate(prompt.tokens(), [&](uint32_t token) {
        std::cout << llama.tokenizer().Token(token) << std::flush;
    });
    std::cout << std::endl;

    return 0;
}
