from __future__ import annotations

import random
import statistics
import unittest

from core.arrival import PoissonArrivalConfig, PoissonArrivalProcess
from models import DNN, DNNProfileCatalog, Task, profile_from_dnn


def _catalog() -> DNNProfileCatalog:
    first = profile_from_dnn(DNN([Task(1, 10.0)], [], 100, 0), "first")
    second = profile_from_dnn(DNN([Task(1, 20.0)], [], 100, 0), "second")
    return DNNProfileCatalog(
        profiles=(first, second),
        sampling_weights=(0.25, 0.75),
    )


class PoissonArrivalTests(unittest.TestCase):
    def test_same_seed_produces_identical_trace(self) -> None:
        config = PoissonArrivalConfig(lambda_per_slot=1.5, origin_nodes=(2, 3))
        first = PoissonArrivalProcess(_catalog(), config, seed=42).generate_trace(20)
        second = PoissonArrivalProcess(_catalog(), config, seed=42).generate_trace(20)
        self.assertEqual(first, second)

    def test_arrival_rng_does_not_consume_global_random_state(self) -> None:
        random.seed(1234)
        expected = random.random()
        random.seed(1234)
        process = PoissonArrivalProcess(
            _catalog(),
            PoissonArrivalConfig(lambda_per_slot=2.0, origin_nodes=(1,)),
            seed=99,
        )
        process.generate_trace(10)
        self.assertEqual(random.random(), expected)

    def test_generated_requests_reference_profiles_and_current_slot(self) -> None:
        process = PoissonArrivalProcess(
            _catalog(),
            PoissonArrivalConfig(lambda_per_slot=5.0, origin_nodes=(4, 5)),
            seed=7,
        )
        trace = process.generate_trace(5)
        requests = [request for batch in trace.batches for request in batch.requests]
        self.assertTrue(requests)
        self.assertEqual(
            [request.request_id for request in requests],
            list(range(len(requests))),
        )
        for batch in trace.batches:
            for request in batch.requests:
                self.assertEqual(request.arrival_slot, batch.slot)
                self.assertIn(request.profile_id, {"first", "second"})
                self.assertIn(request.initiate_node, {4, 5})

    def test_poisson_mean_and_variance_are_close_to_rate(self) -> None:
        rate = 3.0
        process = PoissonArrivalProcess(
            _catalog(),
            PoissonArrivalConfig(lambda_per_slot=rate, origin_nodes=(1,)),
            seed=20260802,
        )
        counts = [len(process.requests_at(slot)) for slot in range(5000)]
        self.assertAlmostEqual(statistics.fmean(counts), rate, delta=0.15)
        self.assertAlmostEqual(statistics.pvariance(counts), rate, delta=0.25)


if __name__ == "__main__":
    unittest.main()
