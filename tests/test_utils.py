import pytest
import time
import torch
import numpy as np
import random

from utils.helpers import set_seed, get_device_info, format_number, Timer


class TestSetSeed:
    def test_reproducible_random(self):
        set_seed(42)
        a = random.random()
        set_seed(42)
        b = random.random()
        assert a == b

    def test_reproducible_numpy(self):
        set_seed(42)
        a = np.random.rand()
        set_seed(42)
        b = np.random.rand()
        assert a == b

    def test_reproducible_torch(self):
        set_seed(42)
        a = torch.rand(1).item()
        set_seed(42)
        b = torch.rand(1).item()
        assert a == b

    def test_different_seeds_differ(self):
        set_seed(42)
        a = random.random()
        set_seed(99)
        b = random.random()
        assert a != b


class TestGetDeviceInfo:
    def test_returns_dict(self):
        info = get_device_info()
        assert isinstance(info, dict)

    def test_has_device_key(self):
        info = get_device_info()
        assert "device" in info
        assert info["device"] in ("cuda", "mps", "cpu")

    def test_has_gpu_name(self):
        info = get_device_info()
        assert "gpu_name" in info


class TestFormatNumber:
    def test_billions(self):
        assert format_number(7_000_000_000) == "7.00B"

    def test_millions(self):
        assert format_number(10_500_000) == "10.50M"

    def test_thousands(self):
        assert format_number(1_500) == "1.5K"

    def test_small_number(self):
        assert format_number(42) == "42"

    def test_exact_billion(self):
        assert format_number(1_000_000_000) == "1.00B"

    def test_exact_million(self):
        assert format_number(1_000_000) == "1.00M"

    def test_exact_thousand(self):
        assert format_number(1_000) == "1.0K"

    def test_zero(self):
        assert format_number(0) == "0"


class TestTimer:
    def test_measures_time(self):
        with Timer() as t:
            time.sleep(0.1)
        assert t.elapsed >= 0.08

    def test_elapsed_ms(self):
        with Timer() as t:
            time.sleep(0.05)
        assert t.elapsed_ms >= 40

    def test_named_timer(self, capsys):
        with Timer("test-op"):
            time.sleep(0.01)
        captured = capsys.readouterr()
        assert "test-op" in captured.out

    def test_elapsed_before_exit_is_none(self):
        t = Timer()
        assert t.elapsed is None
        assert t.elapsed_ms == 0