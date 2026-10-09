#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace exl3 {

class Model {
public:
    explicit Model(const std::string & directory);
    ~Model();
    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;
    size_t tensor_count() const;
    size_t payload_bytes() const;
    size_t hidden_size() const;
    size_t vocab_size() const;
    double bits_per_weight() const;
    std::vector<uint16_t> embedding(uint32_t token) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace exl3
