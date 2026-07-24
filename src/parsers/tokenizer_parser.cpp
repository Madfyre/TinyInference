#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <cctype>
#include <charconv>
#include <fstream>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>


namespace parser {

std::string punctuation = {'!', '.', ',',
                           '?', '/', '\\',
                           '*', '-', '#'
                          };

std::unordered_map<std::string, int32_t> ImportTokens(const std::string& filepath) {
    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("tokenizer not opened!");
    }

    struct stat st;
    fstat(fd, &st);
    
    char* ptr = (char*) mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    madvise(ptr, st.st_size, MADV_SEQUENTIAL);

    std::string_view content{ptr, static_cast<size_t>(st.st_size)};

    size_t vocab_pos = content.find("\"vocab\": {") + 10;
    size_t vocab_end = content.find("merges", vocab_pos);

    std::unordered_map<std::string, int32_t> result;
    result.reserve(128000);

    while (vocab_end > vocab_pos) {
        size_t token_start = content.find("\"", vocab_pos) + 1;
        size_t token_end = content.find("\"", token_start);

        size_t number_start = token_end + 3;
        size_t number_end = content.find(",", number_start);

        int32_t id;
        std::from_chars(content.data() + number_start, content.data() + number_end, id);
        result[{ptr + token_start, ptr + token_end}] = id;
            
        vocab_pos = number_end;
    }

    close(fd);
    munmap(ptr, static_cast<size_t>(st.st_size));

    return result;
}

}