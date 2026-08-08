#include "slam_toolbox/raycast_voxel/engine.hpp"

#include "slam_toolbox/common/voxel_key.hpp"

#include <octomap/OcTree.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <stdexcept>
#include <type_traits>

namespace slam_toolbox::raycast_voxel {
namespace {

using slam_toolbox::common::pack_voxel_key;
using slam_toolbox::common::kKeyBias;
using slam_toolbox::common::kKeyMask;

std::int64_t voxel_coord(double value, double voxel_size) {
    return static_cast<std::int64_t>(std::floor(value / voxel_size));
}

void append_unpacked_key(std::uint64_t key, std::vector<std::int32_t>& output) {
    const auto x = static_cast<std::int64_t>((key >> 42U) & kKeyMask) - kKeyBias;
    const auto y = static_cast<std::int64_t>((key >> 21U) & kKeyMask) - kKeyBias;
    const auto z = static_cast<std::int64_t>(key & kKeyMask) - kKeyBias;
    output.push_back(static_cast<std::int32_t>(x));
    output.push_back(static_cast<std::int32_t>(y));
    output.push_back(static_cast<std::int32_t>(z));
}

template <typename Keys>
std::vector<std::uint64_t> sorted_keys(const Keys& keys) {
    std::vector<std::uint64_t> result;
    result.reserve(keys.size());
    for (const auto& value : keys) {
        if constexpr (std::is_same_v<std::decay_t<decltype(value)>, std::uint64_t>) {
            result.push_back(value);
        } else {
            result.push_back(value.first);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<std::int32_t> unpack_layer(const std::vector<std::uint64_t>& keys) {
    std::vector<std::int32_t> output;
    output.reserve(keys.size() * 3);
    for (const auto key : keys) {
        append_unpacked_key(key, output);
    }
    return output;
}

std::int64_t unpack_key_component(std::uint64_t key, unsigned int shift) {
    return static_cast<std::int64_t>((key >> shift) & kKeyMask) - kKeyBias;
}

}  // namespace

void Config::validate() const {
    if (voxel_size <= 0.0) {
        throw std::invalid_argument("voxel_size must be > 0");
    }
    if (ray_point_stride <= 0) {
        throw std::invalid_argument("ray_point_stride must be > 0");
    }
    if (ray_step_factor <= 0.0) {
        throw std::invalid_argument("ray_step_factor must be > 0");
    }
    if (local_z_min > local_z_max) {
        throw std::invalid_argument("local_z_min must be <= local_z_max");
    }
}

Engine::Engine(Config config) : config_(std::move(config)) {
    config_.validate();
}

bool Engine::point_is_valid(const float* point, bool apply_local_z) const {
    if (!std::isfinite(point[0]) || !std::isfinite(point[1]) ||
        !std::isfinite(point[2]) || !std::isfinite(point[3])) {
        return false;
    }
    const double x = point[0];
    const double y = point[1];
    const double distance_squared = x * x + y * y;
    if (config_.body_radius > 0.0 &&
        distance_squared < config_.body_radius * config_.body_radius) {
        return false;
    }
    if (config_.max_range > 0.0 &&
        distance_squared > config_.max_range * config_.max_range) {
        return false;
    }
    if (apply_local_z &&
        (point[2] < config_.local_z_min || point[2] > config_.local_z_max)) {
        return false;
    }
    return true;
}

Engine::Point3 Engine::transform_point(const float* point, const double* pose) const {
    const double x = static_cast<double>(point[0]) * pose[0] +
                     static_cast<double>(point[1]) * pose[1] +
                     static_cast<double>(point[2]) * pose[2] + pose[3];
    const double y = static_cast<double>(point[0]) * pose[4] +
                     static_cast<double>(point[1]) * pose[5] +
                     static_cast<double>(point[2]) * pose[6] + pose[7];
    const double z = static_cast<double>(point[0]) * pose[8] +
                     static_cast<double>(point[1]) * pose[9] +
                     static_cast<double>(point[2]) * pose[10] + pose[11];
    return {x, y, z};
}

std::uint64_t Engine::point_key(const Point3& point) const {
    return pack_voxel_key(
        voxel_coord(point.x, config_.voxel_size),
        voxel_coord(point.y, config_.voxel_size),
        voxel_coord(point.z, config_.voxel_size));
}

void Engine::trace_free_keys(
    const Point3& origin,
    const Point3& endpoint,
    std::unordered_set<std::uint64_t>& free_keys) const {
    const double dx = endpoint.x - origin.x;
    const double dy = endpoint.y - origin.y;
    const double dz = endpoint.z - origin.z;
    const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double step = std::max(config_.voxel_size * config_.ray_step_factor,
                                 config_.voxel_size * 0.25);
    const double usable = dist - config_.endpoint_margin;
    if (usable <= step || dist <= 0.0) {
        return;
    }
    const auto samples = static_cast<std::int64_t>(std::floor(usable / step));
    std::uint64_t previous_key = 0;
    bool has_previous = false;
    for (std::int64_t sample = 1; sample <= samples; ++sample) {
        const double t = (static_cast<double>(sample) * step) / dist;
        const Point3 point{origin.x + t * dx, origin.y + t * dy, origin.z + t * dz};
        const auto key = point_key(point);
        if (!has_previous || key != previous_key) {
            free_keys.insert(key);
            previous_key = key;
            has_previous = true;
        }
    }
}

void Engine::trace_full_free_keys(
    const Point3& origin,
    const Point3& endpoint,
    std::unordered_set<std::uint64_t>& free_keys) const {
    const double full_dx = endpoint.x - origin.x;
    const double full_dy = endpoint.y - origin.y;
    const double full_dz = endpoint.z - origin.z;
    const double distance = std::sqrt(
        full_dx * full_dx + full_dy * full_dy + full_dz * full_dz);
    const double usable = distance - config_.endpoint_margin;
    if (usable <= 0.0 || distance <= 0.0) {
        return;
    }

    const double scale = usable / distance;
    const Point3 ray_end{
        origin.x + full_dx * scale,
        origin.y + full_dy * scale,
        origin.z + full_dz * scale,
    };
    const double dx = ray_end.x - origin.x;
    const double dy = ray_end.y - origin.y;
    const double dz = ray_end.z - origin.z;

    std::int64_t x = voxel_coord(origin.x, config_.voxel_size);
    std::int64_t y = voxel_coord(origin.y, config_.voxel_size);
    std::int64_t z = voxel_coord(origin.z, config_.voxel_size);
    const std::int64_t end_x = voxel_coord(ray_end.x, config_.voxel_size);
    const std::int64_t end_y = voxel_coord(ray_end.y, config_.voxel_size);
    const std::int64_t end_z = voxel_coord(ray_end.z, config_.voxel_size);

    const double infinity = std::numeric_limits<double>::infinity();
    auto axis_parameters = [this, infinity](
                               double start,
                               double direction,
                               std::int64_t coordinate,
                               std::int64_t& step,
                               double& t_max,
                               double& t_delta) {
        if (direction > 0.0) {
            step = 1;
            const double boundary =
                static_cast<double>(coordinate + 1) * config_.voxel_size;
            t_max = (boundary - start) / direction;
            t_delta = config_.voxel_size / direction;
        } else if (direction < 0.0) {
            step = -1;
            const double boundary = static_cast<double>(coordinate) * config_.voxel_size;
            t_max = (boundary - start) / direction;
            t_delta = -config_.voxel_size / direction;
        } else {
            step = 0;
            t_max = infinity;
            t_delta = infinity;
        }
    };

    std::int64_t step_x = 0;
    std::int64_t step_y = 0;
    std::int64_t step_z = 0;
    double t_max_x = infinity;
    double t_max_y = infinity;
    double t_max_z = infinity;
    double t_delta_x = infinity;
    double t_delta_y = infinity;
    double t_delta_z = infinity;
    axis_parameters(origin.x, dx, x, step_x, t_max_x, t_delta_x);
    axis_parameters(origin.y, dy, y, step_y, t_max_y, t_delta_y);
    axis_parameters(origin.z, dz, z, step_z, t_max_z, t_delta_z);

    free_keys.insert(pack_voxel_key(x, y, z));
    const std::uint64_t maximum_steps =
        static_cast<std::uint64_t>(std::llabs(end_x - x)) +
        static_cast<std::uint64_t>(std::llabs(end_y - y)) +
        static_cast<std::uint64_t>(std::llabs(end_z - z)) + 1U;
    for (std::uint64_t count = 0;
         count < maximum_steps && (x != end_x || y != end_y || z != end_z);
         ++count) {
        if (t_max_x <= t_max_y && t_max_x <= t_max_z) {
            x += step_x;
            t_max_x += t_delta_x;
        } else if (t_max_y <= t_max_z) {
            y += step_y;
            t_max_y += t_delta_y;
        } else {
            z += step_z;
            t_max_z += t_delta_z;
        }
        free_keys.insert(pack_voxel_key(x, y, z));
    }
}

FrameStatistics Engine::integrate_frame(
    const float* scan_xyzi,
    std::size_t point_count,
    const double* pose) {
    if (finalized_) {
        throw std::logic_error("cannot integrate frames after finalize");
    }

    std::vector<Point3> points;
    points.reserve(point_count);
    std::map<std::uint64_t, std::uint64_t> hit_counts;
    FrameStatistics frame;

    for (std::size_t index = 0; index < point_count; ++index) {
        const float* point = scan_xyzi + index * 4;
        if (!point_is_valid(point, true)) {
            continue;
        }
        const auto global = transform_point(point, pose);
        points.push_back(global);
        ++hit_counts[point_key(global)];
    }

    frame.input_points = points.size();
    frame.hit_voxels = hit_counts.size();
    const Point3 origin{pose[3], pose[7], pose[11]};
    std::unordered_set<std::uint64_t> frame_free_keys;
    std::unordered_set<std::uint64_t> frame_occupancy_free_keys;
    for (std::size_t index = 0; index < points.size();
         index += static_cast<std::size_t>(config_.ray_point_stride)) {
        trace_free_keys(origin, points[index], frame_free_keys);
        if (config_.track_full_free_space) {
            trace_full_free_keys(origin, points[index], frame_occupancy_free_keys);
        }
        ++frame.raycast_sample_points;
    }
    for (const auto& [key, count] : hit_counts) {
        static_cast<void>(count);
        frame_free_keys.erase(key);
        frame_occupancy_free_keys.erase(key);
    }

    for (const auto key : frame_free_keys) {
        auto iterator = leaves_.find(key);
        if (iterator == leaves_.end()) {
            continue;
        }
        iterator->second.log_odds -= config_.miss_log_odds;
        ++iterator->second.miss_frames;
        ++frame.free_updates;
    }
    for (const auto& [key, count] : hit_counts) {
        auto& state = leaves_[key];
        state.log_odds += config_.hit_log_odds;
        ++state.hit_frames;
        state.hit_points += count;
    }
    if (config_.track_full_free_space) {
        for (const auto key : frame_occupancy_free_keys) {
            occupancy_log_odds_[key] -= config_.miss_log_odds;
        }
        for (const auto& [key, count] : hit_counts) {
            static_cast<void>(count);
            occupancy_log_odds_[key] += config_.hit_log_odds;
        }
    }

    statistics_.input_points += frame.input_points;
    statistics_.raycast_sample_points += frame.raycast_sample_points;
    return frame;
}

void Engine::finalize() {
    if (finalized_) {
        return;
    }
    occupied_keys_.clear();
    free_keys_.clear();
    kept_keys_.clear();
    for (const auto& [key, state] : leaves_) {
        if (state.log_odds >= config_.occupied_threshold) {
            occupied_keys_.insert(key);
        } else if (state.log_odds <= config_.free_threshold) {
            free_keys_.insert(key);
        }
    }
    const auto unknown_count = leaves_.size() - occupied_keys_.size() - free_keys_.size();
    kept_keys_ = occupied_keys_;
    if (config_.keep_unknown) {
        for (const auto& [key, state] : leaves_) {
            static_cast<void>(state);
            if (occupied_keys_.count(key) == 0 && free_keys_.count(key) == 0) {
                kept_keys_.insert(key);
            }
        }
    }
    statistics_.octree_leaf_voxels = leaves_.size();
    statistics_.occupied_voxels = occupied_keys_.size();
    statistics_.free_voxels = free_keys_.size();
    statistics_.unknown_voxels = unknown_count;
    statistics_.kept_voxels = kept_keys_.size();
    statistics_.removed_voxels =
        config_.keep_unknown ? free_keys_.size() : free_keys_.size() + unknown_count;
    finalized_ = true;
}

ClassifiedFrame Engine::classify_frame(
    const float* scan_xyzi,
    std::size_t point_count,
    const double* pose) const {
    if (!finalized_) {
        throw std::logic_error("finalize must be called before classify_frame");
    }
    ClassifiedFrame output;
    output.points_xyzi.reserve(point_count * 4);
    output.keep_mask.reserve(point_count);

    for (std::size_t index = 0; index < point_count; ++index) {
        const float* point = scan_xyzi + index * 4;
        if (!point_is_valid(point, false)) {
            continue;
        }
        const auto global = transform_point(point, pose);
        const auto key = point_key(global);
        bool keep = false;
        if (config_.keep_unknown) {
            keep = occupied_keys_.count(key) != 0 || free_keys_.count(key) == 0;
        } else {
            keep = occupied_keys_.count(key) != 0;
        }
        const bool outside_roi = point[2] < config_.local_z_min || point[2] > config_.local_z_max;
        const bool ground_protected = config_.ground_protection_enabled &&
                                      point[2] <= config_.ground_protect_local_z_max;
        keep = keep || outside_roi || ground_protected;

        output.points_xyzi.push_back(static_cast<float>(global.x));
        output.points_xyzi.push_back(static_cast<float>(global.y));
        output.points_xyzi.push_back(static_cast<float>(global.z));
        output.points_xyzi.push_back(point[3]);
        output.keep_mask.push_back(keep ? 1 : 0);
    }
    return output;
}

Statistics Engine::statistics() const {
    Statistics result = statistics_;
    result.occupancy_observed_voxels = occupancy_log_odds_.size();
    if (!finalized_) {
        result.octree_leaf_voxels = leaves_.size();
    }
    return result;
}

VoxelLayers Engine::voxel_layers() const {
    if (!finalized_) {
        throw std::logic_error("finalize must be called before voxel_layers");
    }

    std::vector<std::uint64_t> occupied;
    std::vector<std::uint64_t> free;
    std::vector<std::uint64_t> unknown;
    if (config_.track_full_free_space) {
        occupied.reserve(occupancy_log_odds_.size());
        free.reserve(occupancy_log_odds_.size());
        unknown.reserve(occupancy_log_odds_.size());
        for (const auto& [key, log_odds] : occupancy_log_odds_) {
            if (log_odds >= config_.occupied_threshold) {
                occupied.push_back(key);
            } else if (log_odds <= config_.free_threshold) {
                free.push_back(key);
            } else {
                unknown.push_back(key);
            }
        }
        std::sort(occupied.begin(), occupied.end());
        std::sort(free.begin(), free.end());
        std::sort(unknown.begin(), unknown.end());
    } else {
        occupied = sorted_keys(occupied_keys_);
        free = sorted_keys(free_keys_);
        unknown.reserve(statistics_.unknown_voxels);
        for (const auto& [key, state] : leaves_) {
            static_cast<void>(state);
            if (occupied_keys_.count(key) == 0 && free_keys_.count(key) == 0) {
                unknown.push_back(key);
            }
        }
        std::sort(unknown.begin(), unknown.end());
    }
    const auto kept = sorted_keys(kept_keys_);

    std::vector<std::uint64_t> removed;
    removed.reserve(statistics_.removed_voxels);
    for (const auto& [key, state] : leaves_) {
        static_cast<void>(state);
        if (kept_keys_.count(key) == 0) {
            removed.push_back(key);
        }
    }
    std::sort(removed.begin(), removed.end());

    VoxelLayers layers;
    layers.occupied_xyz = unpack_layer(occupied);
    layers.free_xyz = unpack_layer(free);
    layers.unknown_xyz = unpack_layer(unknown);
    layers.kept_xyz = unpack_layer(kept);
    layers.removed_xyz = unpack_layer(removed);
    return layers;
}

OctomapExportStatistics Engine::write_octomap(
    const std::string& binary_path,
    const std::string& full_path) const {
    if (!finalized_) {
        throw std::logic_error("finalize must be called before write_octomap");
    }
    if (binary_path.empty() || full_path.empty()) {
        throw std::invalid_argument("OctoMap output paths must not be empty");
    }

    octomap::OcTree tree(config_.voxel_size);
    OctomapExportStatistics result;
    auto add_node = [this, &tree](std::uint64_t key, double log_odds) {
        const double x =
            (static_cast<double>(unpack_key_component(key, 42U)) + 0.5) * config_.voxel_size;
        const double y =
            (static_cast<double>(unpack_key_component(key, 21U)) + 0.5) * config_.voxel_size;
        const double z =
            (static_cast<double>(unpack_key_component(key, 0U)) + 0.5) * config_.voxel_size;
        if (tree.setNodeValue(x, y, z, static_cast<float>(log_odds), true) == nullptr) {
            throw std::runtime_error(
                "voxel coordinate is outside the standard OctoMap key range");
        }
    };
    if (config_.track_full_free_space) {
        for (const auto& [key, log_odds] : occupancy_log_odds_) {
            if (log_odds >= config_.occupied_threshold) {
                ++result.source_occupied_voxels;
                add_node(key, log_odds);
            } else if (log_odds <= config_.free_threshold) {
                ++result.source_free_voxels;
                add_node(key, log_odds);
            }
        }
    } else {
        for (const auto& [key, state] : leaves_) {
            if (occupied_keys_.count(key) == 0 && free_keys_.count(key) == 0) {
                continue;
            }
            add_node(key, state.log_odds);
        }
        result.source_occupied_voxels = occupied_keys_.size();
        result.source_free_voxels = free_keys_.size();
    }
    tree.updateInnerOccupancy();

    result.tree_nodes = tree.size();
    result.tree_leaf_nodes = tree.getNumLeafNodes();

    // The full tree must be written first: writeBinary() converts the tree to
    // maximum likelihood and prunes it in place.
    if (!tree.write(full_path)) {
        throw std::runtime_error("failed to write full OctoMap file: " + full_path);
    }
    if (!tree.writeBinary(binary_path)) {
        throw std::runtime_error("failed to write binary OctoMap file: " + binary_path);
    }
    return result;
}

}  // namespace slam_toolbox::raycast_voxel
