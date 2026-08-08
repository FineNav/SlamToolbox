#pragma once

#include <cstdint>
#include <stdexcept>

namespace slam_toolbox::common {

constexpr std::int64_t kKeyBits = 21;
constexpr std::int64_t kKeyBias = std::int64_t{1} << (kKeyBits - 1);
constexpr std::int64_t kKeyMask = (std::int64_t{1} << kKeyBits) - 1;

inline std::uint64_t pack_voxel_key(std::int64_t x, std::int64_t y, std::int64_t z) {
    const auto sx = x + kKeyBias;
    const auto sy = y + kKeyBias;
    const auto sz = z + kKeyBias;
    if (sx < 0 || sy < 0 || sz < 0 || sx > kKeyMask || sy > kKeyMask || sz > kKeyMask) {
        throw std::out_of_range("voxel coordinate exceeds the 21-bit packed-key range");
    }
    return (static_cast<std::uint64_t>(sx) << 42U) |
           (static_cast<std::uint64_t>(sy) << 21U) |
           static_cast<std::uint64_t>(sz);
}

}  // namespace slam_toolbox::common
