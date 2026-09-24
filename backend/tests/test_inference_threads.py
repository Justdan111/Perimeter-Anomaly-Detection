"""How many CPU threads inference uses inside a CPU-limited container.

Found on Day 5: ultralytics sets PyTorch's thread count to
`os.cpu_count() - 1` during the first prediction. Inside a container that
is the HOST's core count, not the container's CPU allowance, so a container
limited to 1 CPU ran 7 inference threads fighting over it. Measured in the
Docker image, per frame:

    --cpus=1    7 threads: ~1,050 ms   1 thread (+ OMP_WAIT_POLICY=PASSIVE): ~79 ms
    --cpus=0.5  7 threads: ~3,900 ms   1 thread (+ OMP_WAIT_POLICY=PASSIVE): ~235 ms

These tests cover the pure decision logic: reading the cgroup CPU limit and
choosing a thread count from it. No model, no container needed.
"""

import pytest

from app.services.detector import cpu_limit_from_cgroup, choose_inference_threads


class TestCpuLimitFromCgroup:
    # cgroup v2: /sys/fs/cgroup/cpu.max is "<quota> <period>" or "max <period>".
    @pytest.mark.parametrize(
        "cpu_max,expected",
        [
            ("100000 100000\n", 1.0),
            ("50000 100000\n", 0.5),
            ("10000 100000\n", 0.1),
            ("200000 100000\n", 2.0),
        ],
    )
    def test_v2_quota_is_quota_over_period(self, cpu_max, expected):
        assert cpu_limit_from_cgroup(cpu_max_v2=cpu_max) == pytest.approx(expected)

    def test_v2_max_means_no_limit(self):
        assert cpu_limit_from_cgroup(cpu_max_v2="max 100000\n") is None

    def test_v1_quota_and_period_files(self):
        assert cpu_limit_from_cgroup(cfs_quota_v1="50000\n", cfs_period_v1="100000\n") == 0.5

    def test_v1_minus_one_means_no_limit(self):
        assert cpu_limit_from_cgroup(cfs_quota_v1="-1\n", cfs_period_v1="100000\n") is None

    def test_no_cgroup_files_means_no_limit(self):
        # Native macOS, or a host that exposes neither version.
        assert cpu_limit_from_cgroup() is None

    @pytest.mark.parametrize("garbage", ["", "banana", "100000", "0 100000", "50000 0"])
    def test_unparseable_or_nonsensical_v2_is_treated_as_no_limit(self, garbage):
        # Never crash startup over a diagnostics file; fall back to defaults.
        assert cpu_limit_from_cgroup(cpu_max_v2=garbage) is None


class TestChooseInferenceThreads:
    def test_no_limit_leaves_the_library_default_alone(self):
        # None = don't override. Native runs keep ultralytics' own choice,
        # which measured fastest on the M2 (34 ms/frame).
        assert choose_inference_threads(cpu_limit=None, override=None) is None

    @pytest.mark.parametrize(
        "limit,expected",
        [(0.1, 1), (0.5, 1), (1.0, 1), (1.9, 1), (2.0, 2), (4.0, 4)],
    )
    def test_limited_cpu_uses_whole_cpus_allowed_minimum_one(self, limit, expected):
        # Rounded DOWN: a second thread on 1.9 CPUs gets throttled for the
        # 0.1 it doesn't have, and throttling is what made 7 threads slow.
        assert choose_inference_threads(cpu_limit=limit, override=None) == expected

    def test_explicit_override_wins_over_the_detected_limit(self):
        assert choose_inference_threads(cpu_limit=0.5, override="3") == 3
        assert choose_inference_threads(cpu_limit=None, override="2") == 2

    @pytest.mark.parametrize("bad", ["0", "-1", "two", ""])
    def test_invalid_override_is_an_error_not_silently_ignored(self, bad):
        with pytest.raises(ValueError, match="PERIMETER_TORCH_THREADS"):
            choose_inference_threads(cpu_limit=None, override=bad)
