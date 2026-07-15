from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch

from src.train import setup_distributed


class DistributedSetupTests(unittest.TestCase):
    @unittest.skipIf(torch.cuda.is_available(), "CPU DDP guard test requires a CPU-only runtime")
    def test_cpu_torchrun_is_rejected_by_default(self) -> None:
        environment = {"RANK": "0", "WORLD_SIZE": "2", "LOCAL_RANK": "0"}
        with patch.dict(os.environ, environment, clear=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA를 사용할 수 없습니다"):
                setup_distributed(allow_cpu_ddp=False)


if __name__ == "__main__":
    unittest.main()
