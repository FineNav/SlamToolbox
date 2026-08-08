#include "slam_toolbox/local_hash_voxel/engine.hpp"

#include <cstring>
#include <memory>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using slam_toolbox::local_hash_voxel::ClassifiedFrame;
using slam_toolbox::local_hash_voxel::Config;
using slam_toolbox::local_hash_voxel::Engine;
using slam_toolbox::local_hash_voxel::FrameStatistics;
using slam_toolbox::local_hash_voxel::Statistics;

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
    result["raw_points"] = stats.raw_points;
    result["roi_points"] = stats.roi_points;
    result["endpoint_voxels"] = stats.endpoint_voxels;
    result["traced_rays"] = stats.traced_rays;
    result["visible_updates"] = stats.visible_updates;
    return result;
}

py::dict statistics_dict(const Statistics& stats) {
    py::dict result;
    result["raw_points"] = stats.raw_points;
    result["roi_points"] = stats.roi_points;
    result["endpoint_voxels"] = stats.endpoint_voxels;
    result["traced_rays"] = stats.traced_rays;
    result["visible_updates"] = stats.visible_updates;
    result["stats_voxels"] = stats.stats_voxels;
    result["hit_voxels"] = stats.hit_voxels;
    result["static_voxels"] = stats.static_voxels;
    result["dynamic_voxels"] = stats.dynamic_voxels;
    result["unknown_voxels"] = stats.unknown_voxels;
    return result;
}

}  // namespace

void bind_local_hash_voxel(py::module_& module) {
    auto local_hash = module.def_submodule("local_hash_voxel");

    py::class_<Config>(local_hash, "Config")
        .def(py::init<>())
        .def_readwrite("voxel_size", &Config::voxel_size)
        .def_readwrite("max_range", &Config::max_range)
        .def_readwrite("local_z_min", &Config::local_z_min)
        .def_readwrite("local_z_max", &Config::local_z_max)
        .def_readwrite("body_radius", &Config::body_radius)
        .def_readwrite("ground_protect_local_z_max", &Config::ground_protect_local_z_max)
        .def_readwrite("ground_protection_enabled", &Config::ground_protection_enabled)
        .def_readwrite("lidar_hz", &Config::lidar_hz)
        .def_readwrite("ray_stride", &Config::ray_stride)
        .def_readwrite("max_ray_endpoints", &Config::max_ray_endpoints)
        .def_readwrite("min_visible_frames", &Config::min_visible_frames)
        .def_readwrite("min_visible_time", &Config::min_visible_time)
        .def_readwrite("static_min_hit_ratio", &Config::static_min_hit_ratio)
        .def_readwrite("dynamic_max_hit_ratio", &Config::dynamic_max_hit_ratio)
        .def_readwrite("dynamic_max_hit_time", &Config::dynamic_max_hit_time)
        .def_readwrite("keep_unknown", &Config::keep_unknown)
        .def("validate", &Config::validate);

    py::class_<Engine>(local_hash, "Engine")
        .def(py::init<Config>())
        .def(
            "integrate_frame",
            [](Engine& engine,
               py::array_t<float, py::array::c_style | py::array::forcecast> scan,
               py::array_t<float, py::array::c_style | py::array::forcecast> pose,
               double timestamp) {
                require_scan_shape(scan);
                require_pose_shape(pose);
                FrameStatistics result;
                {
                    py::gil_scoped_release release;
                    result = engine.integrate_frame(
                        scan.data(), static_cast<std::size_t>(scan.shape(0)), pose.data(), timestamp);
                }
                return frame_statistics_dict(result);
            },
            py::arg("scan_xyzi"), py::arg("pose"), py::arg("timestamp"))
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
               py::array_t<float, py::array::c_style | py::array::forcecast> pose) {
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
        });
}
