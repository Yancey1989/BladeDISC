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

#include "triton_disc/TritonGPUToLLVMTranslation.h"

#include "llvm/Support/Debug.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM//ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
//#include "triton/Target/LLVMIR/LLVMIRTranslation.h"
#include "mlir/disc/transforms/register_passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "triton_disc/Conversion/Passes.h"
#include "triton_disc/Conversion/TritonGPUToMLIRGPU.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton_disc;

#define GEN_PASS_CLASSES
#include "triton_disc/Conversion/Passes.h.inc"

using namespace std;
namespace mlir {
namespace triton_disc {

std::unique_ptr<mlir::ModuleOp> translateTritonGPUToLLVMIR(
    llvm::LLVMContext* llvmContext, mlir::ModuleOp module) {
  mlir::PassManager pm(module->getContext());
  applyPassManagerCLOptions(pm);
  auto printingFlags = mlir::OpPrintingFlags();
  printingFlags.elideLargeElementsAttrs(16);
  pm.enableIRPrinting(
      /*shouldPrintBeforePass=*/nullptr,
      /*shouldPrintAfterPass=*/
      [](mlir::Pass* pass, mlir::Operation*) {
        return true;
        // return ::triton::tools::getBoolEnv("MLIR_ENABLE_DUMP");
      },
      /*printModuleScope=*/false,
      /*printAfterOnlyOnChange=*/false,
      /*printAfterOnlyOnFailure*/ false, llvm::dbgs(), printingFlags);

  pm.addPass(mlir::triton_disc::createTritonGPUToMLIRGPUPass());
  //  Canonicalize to eliminate the remaining UnrealizedConversionCastOp
  // pm.addPass(mlir::createCanonicalizerPass());
  // pm.addPass(mlir::createCSEPass());  // Simplify the IR to improve
  // readability. pm.addPass(mlir::createSymbolDCEPass());
  // pm.addPass(mlir::createCanonicalizerPass());

  pm.addPass(mlir::createGpuMapParallelLoopsPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createParallelLoopToGpuPass());
  pm.addPass(mlir::createGpuKernelOutliningPass());
  pm.addPass(::mlir::createLowerAffinePass());
  auto& kernelPm = pm.nest<mlir::gpu::GPUModuleOp>();
  kernelPm.addPass(mlir::disc_ral::createDiscLowerGpuOpsToNVVMOpsPass());
  kernelPm.addPass(mlir::disc_ral::CreateDiscGpuKernelToBlobPass(8, 0));
  if (failed(pm.run(module))) {
    llvm::errs() << "Pass execution failed";
    return nullptr;
  }
  return std::make_unique<mlir::ModuleOp>(module);
}

}  //  namespace triton_disc
}  //  namespace mlir