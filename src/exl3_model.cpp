#include "exl3_model.h"
#include "exl3_codec.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <list>
#include <mutex>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

namespace exl3 {

using json = nlohmann::json;

struct Mapping {
    int fd = -1;
    size_t size = 0;
    const uint8_t * data = nullptr;
    explicit Mapping(const std::string & path) {
        fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("Cannot open model weights");
        struct stat statbuf{};
        if (fstat(fd, &statbuf) || statbuf.st_size < 8) {
            close(fd); fd = -1;
            throw std::runtime_error("Invalid weight file size");
        }
        size = size_t(statbuf.st_size);
        auto mapped = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) {
            close(fd); fd = -1;
            throw std::runtime_error("Cannot map model weights");
        }
        data = static_cast<const uint8_t *>(mapped);
    }
    ~Mapping() { if (data) munmap(const_cast<uint8_t *>(data), size); if (fd >= 0) close(fd); }
};

static size_t positive_integer(const json & value, bool zero_allowed = false) {
    if (!value.is_number_integer()) throw std::runtime_error("Expected an integer dimension or offset");
    const int64_t number = value.get<int64_t>();
    if (number < 0 || (!number && !zero_allowed)) throw std::runtime_error("Invalid dimension or offset");
    return size_t(number);
}

struct Tensor {
    std::string dtype;
    std::vector<size_t> shape;
    const uint8_t * data;
    size_t size;
    const uint16_t * half_data() const {
        if (reinterpret_cast<uintptr_t>(data) % 2) throw std::runtime_error("Unaligned half tensor");
        return reinterpret_cast<const uint16_t *>(data);
    }
};

struct Model::Impl {
    Mapping weights;
    std::unordered_map<std::string, Tensor> tensors;
    size_t payload = 0, k = 0, n = 0, parameters = 0;
    unsigned head_bits = 0;
    Codebook head_codebook = Codebook::three_inst;
    std::string head_prefix;
    mutable std::mutex cache_mutex;
    mutable std::list<std::pair<size_t, std::shared_ptr<std::vector<uint16_t>>>> cache;

    const Tensor & tensor(const std::string & key) const {
        auto found = tensors.find(key);
        if (found == tensors.end()) throw std::runtime_error("Missing model tensor: " + key);
        return found->second;
    }

    explicit Impl(const std::string & directory) : weights(directory + "/model.safetensors") {
        uint64_t header_size = 0;
        for (unsigned i = 0; i < 8; ++i) header_size |= uint64_t(weights.data[i]) << (i * 8);
        if (header_size > 64 * 1024 * 1024 || header_size > weights.size - 8) throw std::runtime_error("Invalid safetensors header length");
        const auto header = json::parse(weights.data + 8, weights.data + 8 + header_size);
        const size_t data_begin = 8 + size_t(header_size);
        std::vector<std::pair<size_t, size_t>> spans;
        const std::unordered_map<std::string, size_t> widths = {{"F16", 2}, {"BF16", 2}, {"I16", 2}, {"F32", 4}, {"I32", 4}};
        for (auto it = header.begin(); it != header.end(); ++it) {
            if (it.key() == "__metadata__") continue;
            const auto & info = it.value();
            Tensor t;
            t.dtype = info.at("dtype").get<std::string>();
            auto width = widths.find(t.dtype);
            if (width == widths.end()) throw std::runtime_error("Unsupported model dtype");
            size_t elements = 1;
            for (const auto & dimension : info.at("shape")) {
                const size_t dim = positive_integer(dimension);
                if (elements > std::numeric_limits<size_t>::max() / dim) throw std::runtime_error("Tensor shape overflow");
                elements *= dim; t.shape.push_back(dim);
            }
            const auto & offsets = info.at("data_offsets");
            if (!offsets.is_array() || offsets.size() != 2) throw std::runtime_error("Invalid tensor offset pair");
            const size_t begin = positive_integer(offsets[0], true), end = positive_integer(offsets[1], true);
            if (end < begin || end > weights.size - data_begin || elements > std::numeric_limits<size_t>::max() / width->second ||
                end - begin != elements * width->second) throw std::runtime_error("Invalid tensor data span");
            t.data = weights.data + data_begin + begin;
            t.size = end - begin;
            tensors.emplace(it.key(), std::move(t));
            spans.emplace_back(begin, end);
        }
        std::sort(spans.begin(), spans.end());
        for (auto span : spans) {
            if (span.first != payload) throw std::runtime_error("Overlapping or missing tensor data");
            payload = span.second;
        }
        if (payload != weights.size - data_begin) throw std::runtime_error("Unaccounted model payload");
        std::ifstream manifest_file(directory + "/hexagon-exl3.json");
        json manifest; manifest_file >> manifest;
        if (manifest.at("format") != "hexagon-exl3" || manifest.at("version") != 1 || manifest.at("kv_cache") != "f16") {
            throw std::runtime_error("Unsupported deployment manifest");
        }
        const auto & embedding = manifest.at("input_embedding");
        if (embedding.at("type") != "tied_linear" || embedding.at("basis") != "original" || embedding.at("token_axis") != 1) {
            throw std::runtime_error("Unsupported embedding alias");
        }
        k = positive_integer(embedding.at("hidden_size"));
        n = positive_integer(embedding.at("vocab_size"));
        if (k % 128 || n % 128 || k > 16384 || n > 1048576) throw std::runtime_error("Invalid embedding geometry");
        parameters = positive_integer(manifest.at("reference_parameter_count"));
        if (positive_integer(manifest.at("weight_payload_bytes")) != payload || double(payload) * 8 / parameters > 5.0) {
            throw std::runtime_error("Model payload or budget mismatch");
        }
        head_prefix = embedding.at("source").get<std::string>();
        const auto & packed = tensor(head_prefix + ".trellis");
        if (packed.dtype != "I16" || packed.shape.size() != 3 || packed.shape[0] != k / 16 || packed.shape[1] != n / 16 ||
            packed.shape[2] % 16 || packed.shape[2] < 16 || packed.shape[2] > 128) throw std::runtime_error("Invalid head trellis");
        head_bits = unsigned(packed.shape[2] / 16);
        for (const auto & entry : {std::make_pair("suh", k), std::make_pair("svh", n)}) {
            const auto & scale = tensor(head_prefix + "." + entry.first);
            if (scale.dtype != "F16" || scale.shape != std::vector<size_t>{entry.second}) throw std::runtime_error("Invalid head scale");
        }
        auto marker = tensors.find(head_prefix + ".mul1");
        if (marker != tensors.end() && tensors.count(head_prefix + ".mcg")) throw std::runtime_error("Ambiguous head codebook");
        if (marker != tensors.end()) {
            uint32_t value;
            if (marker->second.dtype != "I32" || marker->second.size != 4) throw std::runtime_error("Invalid codebook marker");
            std::memcpy(&value, marker->second.data, 4);
            if (value != 0x83dcd12du) throw std::runtime_error("Unsupported mul1 codebook multiplier");
            head_codebook = Codebook::mul1;
        } else if (tensors.count(head_prefix + ".mcg")) {
            const auto & mcg = tensor(head_prefix + ".mcg");
            uint32_t value;
            if (mcg.dtype != "I32" || mcg.size != 4) throw std::runtime_error("Invalid codebook marker");
            std::memcpy(&value, mcg.data, 4);
            if (value != 0xcbac1fedu) throw std::runtime_error("Unsupported mcg codebook multiplier");
            head_codebook = Codebook::mcg;
        }
    }
};

Model::Model(const std::string & directory) : impl(std::make_unique<Impl>(directory)) {}
Model::~Model() = default;
size_t Model::tensor_count() const { return impl->tensors.size(); }
size_t Model::payload_bytes() const { return impl->payload; }
size_t Model::hidden_size() const { return impl->k; }
size_t Model::vocab_size() const { return impl->n; }
double Model::bits_per_weight() const { return double(impl->payload) * 8 / impl->parameters; }

std::vector<uint16_t> Model::embedding(uint32_t token) const {
    if (token >= impl->n) throw std::invalid_argument("Token ID outside vocabulary");
    const size_t group = token / 128;
    std::lock_guard<std::mutex> guard(impl->cache_mutex);
    auto entry = std::find_if(impl->cache.begin(), impl->cache.end(), [&](const auto & item) { return item.first == group; });
    if (entry == impl->cache.end()) {
        auto block = std::make_shared<std::vector<uint16_t>>(impl->k * 128);
        reconstruct_slice(impl->tensor(impl->head_prefix + ".trellis").half_data(),
            impl->tensor(impl->head_prefix + ".suh").half_data(), impl->tensor(impl->head_prefix + ".svh").half_data(),
            impl->k, impl->n, group * 128, 128, impl->head_bits, impl->head_codebook, block->data());
        impl->cache.emplace_front(group, std::move(block));
        if (impl->cache.size() > 16) impl->cache.pop_back();
    } else {
        impl->cache.splice(impl->cache.begin(), impl->cache, entry);
    }
    const auto & block = *impl->cache.front().second;
    std::vector<uint16_t> result(impl->k);
    for (size_t row = 0; row < impl->k; ++row) result[row] = block[row * 128 + token % 128];
    return result;
}

} // namespace exl3
