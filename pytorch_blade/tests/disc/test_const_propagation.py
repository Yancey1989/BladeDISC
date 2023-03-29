# Copyright 2023 The BladeDISC Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import torch
from torch_blade import mlir, jit_pass_constant_propagation

class TestConstantPropagation(unittest.TestCase):
    def test_constant_propagation(self):
        gstr = """
graph(%p1 : Float(1, 512, strides=[512, 1], device=cpu)):
    %1 : int = prim::Constant[value=0]()
    %2 : int = prim::Constant[value=1]()
    %5 : int[] = prim::ListConstruct(%1, %2)
    %permute_1.1 : Tensor = aten::permute(%p1, %5)
    return (%permute_1.1)
        """
        graph = torch._C.parse_ir(gstr)
        #torch._C._jit_pass_constant_propagation(graph)
        jit_pass_constant_propagation(graph)
        print(graph)
    


if __name__ == "__main__":
    unittest.main()