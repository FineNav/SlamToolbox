#include "slam_toolbox/raycast_voxel/engine.hpp"

#include <cstring>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using slam_toolbox::raycast_voxel::ClassifiedFrame;
using slam_toolbox::raycast_voxel::Config;
using slam_toolbox::raycast_voxel::Engine;
using slam_toolbox::raycast_voxel::FrameStatistics;
using slam_toolbox::raycast_voxel::OctomapExportStatistics;
using slam_toolbox::raycast_voxel::Statistics;
using slam_toolbox::raycast_voxel::VoxelLayers;

namespace {

void require_scan_shape(const py::array& scan) {
    if (scan.ndim() != 2 || scan.shape(1) != 4) {
        throw std::invalid_argument("scan_xyzi must have shape (N, 4)");
    }
}

void require_pose_shape(const py::array& pose) {
    if (pose.ndim() != 2 || pose.shape(0) != 4 || pose.shape(1) != 4) {
        throw std::invalid_argument("pose must have shape (4, 4)");
    }
}

py::dict frame_statistics_dict(const FrameStatistics& stats) {
    py::dict result;
    result["input_points"] = stats.input_points;
    result["raycast_sample_points"] = stats.raycast_sample_points;
    result["free_updates"] = stats.free_updates;
    result["hit_voxels"] = stats.hit_voxels;
    return result;
}

py::dict statistics_dict(const Statistics& stats) {
    py::dict result;
    result["octree_leaf_voxels"] = stats.octree_leaf_voxels;
    result["kept_voxels"] = stats.kept_voxels;
    result["occupied_voxels"] = stats.occupied_voxels;
    result["free_voxels"] = stats.free_voxels;
    result["removed_voxels"] = stats.removed_voxels;
    result["unknown_voxels"] = stats.unknown_voxels;
    result["input_points"] = stats.input_points;
    result["raycast_sample_points"] = stats.raycast_sample_points;
    result["occupancy_observed_voxels"] = stats.occupancy_observed_voxels;
    return result;
}

py::dict octomap_statistics_dict(const OctomapExportStatistics& stats) {
    py::dict result;
    result["source_occupied_voxels"] = stats.source_occupied_voxels;
    result["source_free_voxels"] = stats.source_free_voxels;
    result["tree_nodes"] = stats.tree_nodes;
    result["tree_leaf_nodes"] = stats.tree_leaf_nodes;
    return result;
}

py::array_t<std::int32_t> coords_array(std::vector<std::int32_t>&& coords) {
    const auto point_count = static_cast<py::ssize_t>(coords.size() / 3);
    py::array_t<std::int32_t> result({point_count, py::ssize_t{3}});
    if (!coords.empty()) {
        std::memcpy(result.mutable_data(), coords.data(), coords.size() * sizeof(std::int32_t));
    }
    return result;
}

}  // namespace

void bind_raycast_voxel(py::module_& module) {
    auto raycast = module.def_submodule("raycast_voxel");

    py::class_<Config>(raycast, "Config")
        .def(py::init<>())
        .def_readwrite("voxel_size", &Config::voxel_size)
        .def_readwrite("max_range", &Config::max_range)
        .def_readwrite("body_radius", &Config::body_radius)
        .def_readwrite("local_z_min", &Config::local_z_min)
        .def_readwrite("local_z_max", &Config::local_z_max)
        .def_readwrite("ground_protect_local_z_max", &Config::ground_protect_local_z_max)
        .def_readwrite("ground_protection_enabled", &Config::ground_protection_enabled)
        .def_readwrite("ray_point_stride", &Config::ray_point_stride)
        .def_readwrite("ray_step_factor", &Config::ray_step_factor)
        .def_readwrite("endpoint_margin", &Config::endpoint_margin)
        .def_readwrite("hit_log_odds", &Config::hit_log_odds)
        .def_readwrite("miss_log_odds", &Config::miss_log_odds)
        .def_readwrite("occupied_threshold", &Config::occupied_threshold)
        .def_readwrite("free_threshold", &Config::free_threshold)
        .def_readwrite("keep_unknown", &Config::keep_unknown)
        .def_readwrite("track_full_free_space", &Config::track_full_free_space)
        .def("validate", &Config::validate);

    py::class_<Engine>(raycast, "Engine")
        .def(py::init<Config>())
        .def(
            "integrate_frame",
            [](Engine& engine,
               py::array_t<float, py::array::c_style | py::array::forcecast> scan,
               py::array_t<double, py::array::c_style | py::array::forcecast> pose) {
                require_scan_shape(scan);
                require_pose_shape(pose);
                FrameStatistics result;
                {
                    py::gil_scoped_release release;
                    result = engine.integrate_frame(
                        scan.data(), static_cast<std::size_t>(scan.shape(0)), pose.data());
                }
                return frame_statistics_dict(result);
            },
            py::arg("scan_xyzi"), py::arg("pose"))
        .def(
            "finalize",
            [](Engine& engine) {
                py::gil_scoped_release release;
                engine.finalize();
            })
        .def(
            "classify_frame",
            [](const Engine& engine,
               py::array_t<float, py::array::c_style | py::array::forcecast> scan,
               py::array_t<double, py::array::c_style | py::array::forcecast> pose) {
                require_scan_shape(scan);
                require_pose_shape(pose);
                ClassifiedFrame result;
                {
                    py::gil_scoped_release release;
                    result = engine.classify_frame(
                        scan.data(), static_cast<std::size_t>(scan.shape(0)), pose.data());
                }
                const auto point_count = static_cast<py::ssize_t>(result.keep_mask.size());
                py::array_t<float> points({point_count, py::ssize_t{4}});
                py::array_t<bool> keep_mask(point_count);
                if (!result.points_xyzi.empty()) {
                    std::memcpy(
                        points.mutable_data(), result.points_xyzi.data(),
                        result.points_xyzi.size() * sizeof(float));
                }
                bool* mask_data = keep_mask.mutable_data();
                for (py::ssize_t index = 0; index < point_count; ++index) {
                    mask_data[index] = result.keep_mask[static_cast<std::size_t>(index)] != 0;
                }
                return py::make_tuple(std::move(points), std::move(keep_mask));
            },
            py::arg("scan_xyzi"), py::arg("pose"))
        .def("statistics", [](const Engine& engine) {
            return statistics_dict(engine.statistics());
        })
        .def("voxel_layers", [](const Engine& engine) {
            VoxelLayers layers;
            {
                py::gil_scoped_release release;
                layers = engine.voxel_layers();
            }
            py::dict result;
            result["occupied"] = coords_array(std::move(layers.occupied_xyz));
            result["free"] = coords_array(std::move(layers.free_xyz));
            result["unknown"] = coords_array(std::move(layers.unknown_xyz));
            result["kept"] = coords_array(std::move(layers.kept_xyz));
            result["removed"] = coords_array(std::move(layers.removed_xyz));
            return result;
        })
        .def(
            "write_octomap",
            [](const Engine& engine,
               const std::string& binary_path,
               const std::string& full_path) {
                OctomapExportStatistics stats;
                {
                    py::gil_scoped_release release;
                    stats = engine.write_octomap(binary_path, full_path);
                }
                return octomap_statistics_dict(stats);
            },
            py::arg("binary_path"), py::arg("full_path"));
}
