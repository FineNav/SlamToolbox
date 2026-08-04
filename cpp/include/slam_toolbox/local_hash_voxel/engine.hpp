#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace slam_toolbox::local_hash_voxel {

struct Config {
    double voxel_size = 0.5;
    double max_range = 30.0;
    double local_z_min = 0.0;
    double local_z_max = 3.0;
    double body_radius = 0.5;
    double ground_protect_local_z_max = 0.0;
    bool ground_protection_enabled = false;
    double lidar_hz = 10.0;
    std::int64_t ray_stride = 4;
    std::int64_t max_ray_endpoints = 25000;
    std::int64_t min_visible_frames = 10;
    double min_visible_time = 2.0;
    double static_min_hit_ratio = 0.5;
    double dynamic_max_hit_ratio = 0.15;
    double dynamic_max_hit_time = 2.0;
    bool keep_unknown = true;

    void validate() const;
};

struct FrameStatistics {
    std::uint64_t raw_points = 0;
    std::uint64_t roi_points = 0;
    std::uint64_t endpoint_voxels = 0;
    std::uint64_t traced_rays = 0;
    std::uint64_t visible_updates = 0;
};

struct Statistics {
    std::uint64_t raw_points = 0;
    std::uint64_t roi_points = 0;
    std::uint64_t endpoint_voxels = 0;
    std::uint64_t traced_rays = 0;
    std::uint64_t visible_updates = 0;
    std::uint64_t stats_voxels = 0;
    std::uint64_t hit_voxels = 0;
    std::uint64_t static_voxels = 0;
    std::uint64_t dynamic_voxels = 0;
    std::uint64_t unknown_voxels = 0;
};

struct ClassifiedFrame {
    std::vector<float> points_xyzi;
    std::vector<std::uint8_t> keep_mask;
};

class Engine {
public:
    explicit Engine(Config config);

    FrameStatistics integrate_frame(
        const float* scan_xyzi,
        std::size_t point_count,
        const float* pose,
        double timestamp);
    void finalize();
    ClassifiedFrame classify_frame(
        const float* scan_xyzi,
        std::size_t point_count,
        const float* pose) const;
    Statistics statistics() const;

private:
    struct VoxelStat {
        std::uint64_t visible_frames = 0;
        std::uint64_t hit_frames = 0;
        std::uint64_t hit_count = 0;
        double first_hit_time = 0.0;
        double last_hit_time = 0.0;
        bool has_hit = false;
    };

    struct Point3 {
        double x;
        double y;
        double z;
    };

    bool point_is_valid(const float* point, bool apply_local_z) const;
    Point3 transform_point(const float* point, const float* pose) const;
    std::vector<std::uint64_t> trace_ray(const Point3& origin, const Point3& endpoint) const;

    Config config_;
    std::unordered_map<std::uint64_t, VoxelStat> voxel_stats_;
    std::unordered_set<std::uint64_t> kept_keys_;
    Statistics statistics_;
    bool finalized_ = false;
};

}  // namespace slam_toolbox::local_hash_voxel
