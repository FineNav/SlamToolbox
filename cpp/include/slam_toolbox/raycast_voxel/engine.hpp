#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>

namespace slam_toolbox::raycast_voxel {

struct Config {
    double voxel_size = 0.5;
    double max_range = 30.0;
    double body_radius = 0.5;
    double local_z_min = 0.0;
    double local_z_max = 3.0;
    double ground_protect_local_z_max = 0.0;
    bool ground_protection_enabled = false;
    std::int64_t ray_point_stride = 4;
    double ray_step_factor = 0.75;
    double endpoint_margin = 0.10;
    double hit_log_odds = 0.40;
    double miss_log_odds = 0.80;
    double occupied_threshold = 1.5;
    double free_threshold = -1.0;
    bool keep_unknown = true;
    bool track_full_free_space = false;

    void validate() const;
};

struct FrameStatistics {
    std::uint64_t input_points = 0;
    std::uint64_t raycast_sample_points = 0;
    std::uint64_t free_updates = 0;
    std::uint64_t hit_voxels = 0;
};

struct Statistics {
    std::uint64_t octree_leaf_voxels = 0;
    std::uint64_t kept_voxels = 0;
    std::uint64_t occupied_voxels = 0;
    std::uint64_t free_voxels = 0;
    std::uint64_t removed_voxels = 0;
    std::uint64_t unknown_voxels = 0;
    std::uint64_t input_points = 0;
    std::uint64_t raycast_sample_points = 0;
    std::uint64_t occupancy_observed_voxels = 0;
};

struct VoxelLayers {
    std::vector<std::int32_t> occupied_xyz;
    std::vector<std::int32_t> free_xyz;
    std::vector<std::int32_t> unknown_xyz;
    std::vector<std::int32_t> kept_xyz;
    std::vector<std::int32_t> removed_xyz;
};

struct OctomapExportStatistics {
    std::uint64_t source_occupied_voxels = 0;
    std::uint64_t source_free_voxels = 0;
    std::uint64_t tree_nodes = 0;
    std::uint64_t tree_leaf_nodes = 0;
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
        const double* pose);
    void finalize();
    ClassifiedFrame classify_frame(
        const float* scan_xyzi,
        std::size_t point_count,
        const double* pose) const;
    Statistics statistics() const;
    VoxelLayers voxel_layers() const;
    OctomapExportStatistics write_octomap(
        const std::string& binary_path,
        const std::string& full_path) const;

private:
    struct Point3 {
        double x;
        double y;
        double z;
    };

    struct VoxelState {
        double log_odds = 0.0;
        std::uint64_t hit_frames = 0;
        std::uint64_t miss_frames = 0;
        std::uint64_t hit_points = 0;
    };

    bool point_is_valid(const float* point, bool apply_local_z) const;
    Point3 transform_point(const float* point, const double* pose) const;
    std::uint64_t point_key(const Point3& point) const;
    void trace_free_keys(
        const Point3& origin,
        const Point3& endpoint,
        std::unordered_set<std::uint64_t>& free_keys) const;
    void trace_full_free_keys(
        const Point3& origin,
        const Point3& endpoint,
        std::unordered_set<std::uint64_t>& free_keys) const;

    Config config_;
    std::unordered_map<std::uint64_t, VoxelState> leaves_;
    std::unordered_map<std::uint64_t, double> occupancy_log_odds_;
    std::unordered_set<std::uint64_t> occupied_keys_;
    std::unordered_set<std::uint64_t> free_keys_;
    std::unordered_set<std::uint64_t> kept_keys_;
    Statistics statistics_;
    bool finalized_ = false;
};

}  // namespace slam_toolbox::raycast_voxel
