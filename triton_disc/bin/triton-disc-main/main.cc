// Copyright 2023 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <fstream>

#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/disc/transforms/register_passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton_disc/Conversion/Passes.h"
#include "triton_disc/Conversion/TritonGPUToMLIRGPU.h"
#include "triton_disc/TritonGPUToLLVMTranslation.h"

namespace mlir {
namespace triton_disc {

static mlir::OwningOpRef<mlir::ModuleOp> parseMLIRInput(StringRef inputFilename,
                                                        MLIRContext* context) {
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  return mlir::OwningOpRef<mlir::ModuleOp>(
      parseSourceFile<mlir::ModuleOp>(sourceMgr, context));
}

void DumpCubin(const std::string& fname, mlir::ModuleOp& module) {
  OpBuilder builder(module->getContext());
  SmallVector<Operation*, 1> gpuModules;
  module.walk([&](gpu::GPUModuleOp moduleOp) {
    gpuModules.push_back(builder.clone(*moduleOp.getOperation()));
  });
  auto binary = gpuModules[0]
                    ->getAttrOfType<StringAttr>(
                        StringAttr::get(module->getContext(), "gpu.binary"))
                    .str();

  std::ofstream out("a.out", std::ios::out | std::ios::binary);
  out << binary;
  out.close();
  return;
}

mlir::LogicalResult TritonDISCMain(int argc, char** argv,
                                   DialectRegistry& registry) {
  MLIRContext context(registry);
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  auto m = parseMLIRInput("tests/vec_add.mlir", &context);
  if (!m) {
    llvm::errs() << "could not parse the input file\n";
    return failure();
  }

  mlir::ModuleOp module = m.get();
  auto retModuleOp = translateTritonGPUToLLVMIR(nullptr, module);
  if (!retModuleOp) {
    llvm::errs() << "failed to translate to LLVM IR\n";
    return failure();
  }
  retModuleOp->dump();
  DumpCubin("vec_add.cubin", module);
  return success();
}

}  //  namespace triton_disc
}  //  namespace mlir

int main(int argc, char** argv) {
  mlir::registerAllPasses();
  mlir::registerTritonPasses();
  mlir::disc_ral::registerDiscLowerGpuOpsToNVVMOpsPass();
  mlir::disc_ral::registerGpuKernelToBlobPass();
  mlir::triton::registerConvertTritonToTritonGPUPass();
  mlir::triton_disc::registerConvertTritonGPUToMLIRGPUPass();

  // TODO: register Triton & TritonGPU passes
  mlir::DialectRegistry registry;
  registry.insert<mlir::triton::TritonDialect,
                  mlir::triton::gpu::TritonGPUDialect, mlir::func::FuncDialect,
                  mlir::math::MathDialect, mlir::arith::ArithDialect,
                  mlir::scf::SCFDialect, mlir::gpu::GPUDialect>();

  return mlir::asMainReturnCode(
      mlir::triton_disc::TritonDISCMain(argc, argv, registry));
}
