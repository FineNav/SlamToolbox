#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_local_hash_voxel(py::module_& module);

PYBIND11_MODULE(_native, module) {
    module.doc() = "Native compute backends for slam_toolbox";
    module.attr("api_version") = 1;
    bind_local_hash_voxel(module);
}
