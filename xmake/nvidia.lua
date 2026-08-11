local corex = os.isfile("/usr/local/corex/bin/clang++")

if corex then
    target("llaisys-corex-cudadevrt")
        set_kind("static")
        set_basename("cudadevrt")
        set_default(false)
        set_languages("cxx17")
        add_files("../src/device/nvidia/corex_cudadevrt_stub.cpp")

        on_install(function (target) end)
    target_end()
end

target("llaisys-device-nvidia")
    set_kind("static")
    add_deps("llaisys-utils")
    if corex then
        add_deps("llaisys-corex-cudadevrt")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if corex then
        set_toolset("cu", "/usr/local/corex/bin/clang++")
        add_cuflags("-x", "ivcore", "-std=c++17", "--cuda-path=/usr/local/corex", "--offload-arch=native", {force = true})
        set_values("cuda.rdc", false)
        if not is_plat("windows") then
            add_cuflags("-fPIC")
        end
    else
        if not is_plat("windows") then
            add_cuflags("-Xcompiler=-fPIC")
            add_culdflags("-Xcompiler=-fPIC")
        end
        add_cugencodes("native")
        add_values("cuda.build.devlink", true)
    end

    add_files("../src/device/nvidia/*.cu")
    add_links("cudart")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor", "llaisys-device-nvidia")

    set_languages("cxx17")
    set_warnings("all", "error")
    if corex then
        set_toolset("cu", "/usr/local/corex/bin/clang++")
        add_cuflags("-x", "ivcore", "-std=c++17", "--cuda-path=/usr/local/corex", "--offload-arch=native", {force = true})
        set_values("cuda.rdc", false)
        if not is_plat("windows") then
            add_cuflags("-fPIC")
        end
    else
        if not is_plat("windows") then
            add_cuflags("-Xcompiler=-fPIC")
            add_culdflags("-Xcompiler=-fPIC")
        end
        add_cugencodes("native")
        add_values("cuda.build.devlink", true)
    end

    add_files("../src/ops/*/nvidia/*.cu")
    add_links("cudart", "cublas")

    on_install(function (target) end)
target_end()
