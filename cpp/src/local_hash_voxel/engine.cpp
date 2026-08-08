#include "slam_toolbox/local_hash_voxel/engine.hpp"

#include "slam_toolbox/common/voxel_key.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace slam_toolbox::local_hash_voxel {
namespace {

using slam_toolbox::common::pack_voxel_key;

std::int64_t voxel_coord(double value, double voxel_size) {
    return static_cast<std::int64_t>(std::floor(value / voxel_size));
}

}  // namespace

void Config::validate() const {
    if (voxel_size <= 0.0) {
        throw std::invalid_argument("voxel_size must be > 0");
    }
    if (max_range <= 0.0) {
        throw std::invalid_argument("max_range must be > 0");
    }
    if (local_z_min > local_z_max) {
        throw std::invalid_argument("local_z_min must be <= local_z_max");
    }
    if (lidar_hz <= 0.0) {
        throw std::invalid_argument("lidar_hz must be > 0");
    }
    if (ray_stride <= 0) {
        throw std::invalid_argument("ray_stride must be > 0");
    }
    if (max_ray_endpoints < 0) {
        throw std::invalid_argument("max_ray_endpoints must be >= 0");
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

Engine::Point3 Engine::transform_point(const float* point, const float* pose) const {
    const float x = point[0] * pose[0] + point[1] * pose[1] + point[2] * pose[2] + pose[3];
    const float y = point[0] * pose[4] + point[1] * pose[5] + point[2] * pose[6] + pose[7];
    const float z = point[0] * pose[8] + point[1] * pose[9] + point[2] * pose[10] + pose[11];
    return {x, y, z};
}

std::vector<std::uint64_t> Engine::trace_ray(
    const Point3& origin,
    const Point3& endpoint) const {
    std::array<std::int64_t, 3> start{
        voxel_coord(origin.x, config_.voxel_size),
        voxel_coord(origin.y, config_.voxel_size),
        voxel_coord(origin.z, config_.voxel_size),
    };
    std::array<std::int64_t, 3> end{
        voxel_coord(endpoint.x, config_.voxel_size),
        voxel_coord(endpoint.y, config_.voxel_size),
        voxel_coord(endpoint.z, config_.voxel_size),
    };
    auto current = start;
    const std::array<double, 3> origin_values{origin.x, origin.y, origin.z};
    const std::array<double, 3> direction{
        endpoint.x - origin.x,
        endpoint.y - origin.y,
        endpoint.z - origin.z,
    };
    std::array<std::int64_t, 3> step{};
    std::array<double, 3> t_max{};
    std::array<double, 3> t_delta{};

    for (std::size_t axis = 0; axis < 3; ++axis) {
        step[axis] = direction[axis] > 0.0 ? 1 : (direction[axis] < 0.0 ? -1 : 0);
        if (direction[axis] > 0.0) {
            const double boundary = static_cast<double>(current[axis] + 1) * config_.voxel_size;
            t_max[axis] = (boundary - origin_values[axis]) / direction[axis];
            t_delta[axis] = config_.voxel_size / direction[axis];
        } else if (direction[axis] < 0.0) {
            const double boundary = static_cast<double>(current[axis]) * config_.voxel_size;
            t_max[axis] = (boundary - origin_values[axis]) / direction[axis];
            t_delta[axis] = -config_.voxel_size / direction[axis];
        } else {
            t_max[axis] = std::numeric_limits<double>::infinity();
            t_delta[axis] = std::numeric_limits<double>::infinity();
        }
    }

    const auto max_steps = std::llabs(end[0] - start[0]) +
                           std::llabs(end[1] - start[1]) +
                           std::llabs(end[2] - start[2]) + 1;
    std::vector<std::uint64_t> keys;
    keys.reserve(static_cast<std::size_t>(max_steps + 1));
    for (std::int64_t iteration = 0; iteration <= max_steps; ++iteration) {
        keys.push_back(pack_voxel_key(current[0], current[1], current[2]));
        if (current == end) {
            break;
        }
        std::size_t axis = 0;
        if (t_max[1] < t_max[axis]) {
            axis = 1;
        }
        if (t_max[2] < t_max[axis]) {
            axis = 2;
        }
        current[axis] += step[axis];
        t_max[axis] += t_delta[axis];
    }
    return keys;
}

FrameStatistics Engine::integrate_frame(
    const float* scan_xyzi,
    std::size_t point_count,
    const float* pose,
    double timestamp) {
    if (finalized_) {
        throw std::logic_error("cannot integrate frames after finalize");
    }

    struct Endpoint {
        Point3 point{};
        std::uint64_t count = 0;
    };
    std::map<std::uint64_t, Endpoint> endpoints;
    FrameStatistics frame;
    frame.raw_points = point_count;

    for (std::size_t index = 0; index < point_count; ++index) {
        const float* point = scan_xyzi + index * 4;
        if (!point_is_valid(point, true)) {
            continue;
        }
        const Point3 global = transform_point(point, pose);
        const auto key = pack_voxel_key(
            voxel_coord(global.x, config_.voxel_size),
            voxel_coord(global.y, config_.voxel_size),
            voxel_coord(global.z, config_.voxel_size));
        auto [iterator, inserted] = endpoints.try_emplace(key, Endpoint{global, 0});
        ++iterator->second.count;
        ++frame.roi_points;
    }

    frame.endpoint_voxels = endpoints.size();
    const Point3 origin{pose[3], pose[7], pose[11]};
    std::unordered_set<std::uint64_t> visible_keys;
    std::int64_t endpoint_index = 0;
    for (const auto& [key, endpoint] : endpoints) {
        static_cast<void>(key);
        if (endpoint_index % config_.ray_stride == 0 &&
            (config_.max_ray_endpoints == 0 ||
             frame.traced_rays < static_cast<std::uint64_t>(config_.max_ray_endpoints))) {
            const auto ray_keys = trace_ray(origin, endpoint.point);
            visible_keys.insert(ray_keys.begin(), ray_keys.end());
            ++frame.traced_rays;
        }
        ++endpoint_index;
    }
    frame.visible_updates = visible_keys.size();

    for (const auto key : visible_keys) {
        ++voxel_stats_[key].visible_frames;
    }
    for (const auto& [key, endpoint] : endpoints) {
        auto& stat = voxel_stats_[key];
        ++stat.hit_frames;
        stat.hit_count += endpoint.count;
        if (!stat.has_hit) {
            stat.first_hit_time = timestamp;
            stat.has_hit = true;
        }
        stat.last_hit_time = timestamp;
    }

    statistics_.raw_points += frame.raw_points;
    statistics_.roi_points += frame.roi_points;
    statistics_.endpoint_voxels += frame.endpoint_voxels;
    statistics_.traced_rays += frame.traced_rays;
    statistics_.visible_updates += frame.visible_updates;
    return frame;
}

void Engine::finalize() {
    if (finalized_) {
        return;
    }
    kept_keys_.clear();
    statistics_.stats_voxels = voxel_stats_.size();

    for (const auto& [key, stat] : voxel_stats_) {
        if (!stat.has_hit || stat.hit_frames == 0) {
            continue;
        }
        ++statistics_.hit_voxels;
        const double visible_time = static_cast<double>(stat.visible_frames) / config_.lidar_hz;
        const double hit_time = std::max(0.0, stat.last_hit_time - stat.first_hit_time);
        const double hit_ratio = static_cast<double>(stat.hit_frames) /
                                 static_cast<double>(std::max<std::uint64_t>(1, stat.visible_frames));
        const bool enough_visible =
            stat.visible_frames >= static_cast<std::uint64_t>(std::max<std::int64_t>(0, config_.min_visible_frames)) &&
            visible_time >= config_.min_visible_time;
        if (enough_visible && hit_ratio >= config_.static_min_hit_ratio) {
            kept_keys_.insert(key);
            ++statistics_.static_voxels;
        } else if (
            enough_visible && hit_ratio <= config_.dynamic_max_hit_ratio &&
            hit_time <= config_.dynamic_max_hit_time) {
            ++statistics_.dynamic_voxels;
        } else {
            if (config_.keep_unknown) {
                kept_keys_.insert(key);
            }
            ++statistics_.unknown_voxels;
        }
    }
    finalized_ = true;
}

ClassifiedFrame Engine::classify_frame(
    const float* scan_xyzi,
    std::size_t point_count,
    const float* pose) const {
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
        const Point3 global = transform_point(point, pose);
        const auto key = pack_voxel_key(
            voxel_coord(global.x, config_.voxel_size),
            voxel_coord(global.y, config_.voxel_size),
            voxel_coord(global.z, config_.voxel_size));
        const bool outside_roi = point[2] < config_.local_z_min || point[2] > config_.local_z_max;
        const bool ground_protected = config_.ground_protection_enabled &&
                                      point[2] <= config_.ground_protect_local_z_max;
        const bool keep = outside_roi || ground_protected || kept_keys_.count(key) != 0;
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
    result.stats_voxels = voxel_stats_.size();
    return result;
}

}  // namespace slam_toolbox::local_hash_voxel
