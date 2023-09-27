load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "gentbl_filegroup", "td_library")
load(
    "@org_tensorflow//tensorflow/tsl/platform:build_config.bzl",
    "tf_proto_library",
)
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")

package(
    default_visibility = [
        "//visibility:public",
    ],
)

gentbl_cc_library(
    name = "TritonDISCPassIncGen",
    strip_include_prefix = "",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=triton_disc",
            ],
            "include/triton_disc/Conversion/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "include/triton_disc/Conversion/Passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
        "@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:MemRefDialect",
    ],
)

cc_library(
    name = "tritongpu-to-llvm-translation",
    includes = [
        "include"
    ],
    srcs = [
        "lib/DISC/TritonGPUToLLVMTranslation.cc",
        "lib/Conversion/TritonGPUToMLIRGPU.cc",
    ],
    hdrs = [
        "include/triton_disc/Conversion/Passes.h",
        "include/triton_disc/TritonGPUToLLVMTranslation.h",
        "include/triton_disc/Conversion/TritonGPUToMLIRGPU.h",
        "@org_triton//:cuda_headers",
    ],
    deps = [
        ":TritonDISCPassIncGen",
        "@org_triton//:TritonDialect",
        "@org_triton//:TritonGPUIR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUPassIncGen",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@org_disc_compiler//mlir/disc:disc_gpu_kernel_to_blob",
        "@org_disc_compiler//mlir/disc:all_passes",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "triton-disc-main",
    srcs = [
        "bin/triton-disc-main/main.cc",
        "lib/Conversion/TritonGPUToMLIRGPU.cc",
        "lib/DISC/TritonGPUToLLVMTranslation.cc",
    ],
    deps = [
        ":tritongpu-to-llvm-translation",
        "@org_triton//:TritonGPUIR",
        "@org_triton//:TritonDialectCombine",
        "@org_triton//:TritonConversionPasses",
        "@org_triton//:TritonAnalysis",
        "@org_triton//:TritonDialectsAndPasses",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:AffineDialect",
        "@org_disc_compiler//mlir/disc/tools/disc-transform:all_passes",
        "@org_disc_compiler//mlir/disc:disc_ral",
        "@org_disc_compiler//mlir/disc:all_passes",
        "@org_tensorflow//tensorflow/tsl/platform:env_impl",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:all_passes",
        "@org_tensorflow//tensorflow/tsl/profiler/utils:time_utils_impl",
        "@org_tensorflow//tensorflow/tsl/profiler/backends/cpu:traceme_recorder_impl",
    ],
)

cc_binary(
    name = "triton-disc-opt",
    srcs = [
        "bin/triton_disc_opt.cc",
        "lib/Conversion/TritonGPUToMLIRGPU.cc",
    ],
    deps = [
        ":tritongpu-to-llvm-translation",
        "@org_triton//:TritonGPUIR",
        "@org_triton//:TritonDialectCombine",
        "@org_triton//:TritonConversionPasses",
        "@org_triton//:TritonAnalysis",
        "@org_triton//:TritonDialectsAndPasses",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:AffineDialect",
        "@org_disc_compiler//mlir/disc/tools/disc-transform:all_passes",
        "@org_disc_compiler//mlir/disc:disc_ral",
        "@org_disc_compiler//mlir/disc:all_passes",
        "@org_tensorflow//tensorflow/tsl/platform:env_impl",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:all_passes",
        "@org_tensorflow//tensorflow/tsl/profiler/utils:time_utils_impl",
        "@org_tensorflow//tensorflow/tsl/profiler/backends/cpu:traceme_recorder_impl",
    ],
)

pybind_library(
    name = "triton_disc_pybind11",
    srcs = [
        "python/csrc/triton_disc.cc",
    ],
    deps = [
        ":tritongpu-to-llvm-translation",
        "@com_google_protobuf//:protobuf",
        "@org_triton//:TritonGPUIR",
        "@org_triton//:TritonDialectCombine",
        "@org_triton//:TritonConversionPasses",
        "@org_triton//:TritonAnalysis",
        "@org_triton//:TritonDialectsAndPasses",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:LLVMDialect",
        "@org_disc_compiler//mlir/disc:disc_gpu_kernel_to_blob",
        "@org_disc_compiler//mlir/disc/tools/disc-transform:all_passes",
        "@org_disc_compiler//mlir/disc:disc_ral",
        "@org_disc_compiler//mlir/disc:all_passes",
        "@org_tensorflow//tensorflow/core/framework:tensor_shape",
        "@org_tensorflow//tensorflow/core/framework:resource_handle_proto_cc",
        "@org_tensorflow//tensorflow/core/framework:types_proto_cc",
        "@org_tensorflow//tensorflow/core/framework:log_memory_proto_cc",
        "@org_tensorflow//tensorflow/compiler/xla:xla_proto_cc",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core/framework:tensor",
        "@org_tensorflow//tensorflow/tsl/platform:env_impl",
        "@org_tensorflow//tensorflow/compiler/xla:debug_options_flags",
        "@org_tensorflow//tensorflow/compiler/xla/mlir_hlo:all_passes",
        "@org_tensorflow//tensorflow/compiler/xla:execution_options_util",
        "@org_tensorflow//tensorflow/compiler/xla:xla_data_proto_cc",
        "@org_tensorflow//tensorflow/tsl/protobuf:histogram_proto_cc",
        "@org_tensorflow//tensorflow/tsl/protobuf:dnn_proto_cc",
        "@org_tensorflow//tensorflow/tsl/profiler/utils:time_utils_impl",
        "@org_tensorflow//tensorflow/tsl/profiler/backends/cpu:host_tracer_utils",
        "@org_tensorflow//tensorflow/tsl/profiler/backends/cpu:traceme_recorder_impl",
        "@org_tensorflow//tensorflow/tsl/framework:device_type",
    ],
    linkstatic = 0,
)

pybind_extension(
    name = "_triton_disc",
    linkopts = ["-Wl,-rpath,$$ORIGIN"],
    deps = [
        ":triton_disc_pybind11",
    ]
)
