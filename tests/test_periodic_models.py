from __future__ import annotations

import unittest

from models import (
    DNN,
    DNNProfileCatalog,
    InferenceRequest,
    LinkDNN,
    Task,
    materialize_dnn,
    profile_from_dnn,
)


class PeriodicModelTests(unittest.TestCase):
    def test_profile_round_trip_keeps_dag_and_separates_request_fields(self) -> None:
        first = Task(cpu_need=1, float_num=10.0)
        second = Task(cpu_need=2, float_num=20.0)
        dnn = DNN(
            tasks=[first, second],
            links=[LinkDNN(first, second, 64)],
            delay=900,
            initiateNode=7,
            startFloat=128.0,
            preference_stability=0.8,
            preference_delay=0.1,
            preference_energy=0.1,
            backFloat=32.0,
        )
        profile = profile_from_dnn(dnn, "profile-a")
        request = InferenceRequest(
            request_id=4,
            profile_id="profile-a",
            arrival_slot=3,
            initiate_node=9,
            deadline_ms=700.0,
            preference_stability=0.5,
            preference_delay=0.3,
            preference_energy=0.2,
        )

        materialized = materialize_dnn(profile, request)

        self.assertEqual(len(materialized.tasks), 2)
        self.assertIs(materialized.links[0].s_task, materialized.tasks[0])
        self.assertIs(materialized.links[0].e_task, materialized.tasks[1])
        self.assertEqual(materialized.links[0].float_tran, 64)
        self.assertEqual(materialized.initiateNode, 9)
        self.assertEqual(materialized.delay, 700)
        self.assertEqual(materialized.startFloat, 128.0)
        self.assertEqual(materialized.backFloat, 32.0)
        self.assertEqual(materialized.preference_stability, 0.5)
        self.assertEqual(materialized.preference_delay, 0.3)
        self.assertEqual(materialized.preference_energy, 0.2)

    def test_catalog_rejects_duplicate_profile_ids(self) -> None:
        task = Task(cpu_need=1, float_num=1.0)
        profile = profile_from_dnn(DNN([task], [], 100, 0), "same")
        with self.assertRaisesRegex(ValueError, "unique"):
            DNNProfileCatalog(
                profiles=(profile, profile),
                sampling_weights=(0.5, 0.5),
            )

    def test_request_rejects_invalid_preference(self) -> None:
        with self.assertRaisesRegex(ValueError, "preference_stability"):
            InferenceRequest(
                request_id=0,
                profile_id="p",
                arrival_slot=0,
                initiate_node=0,
                deadline_ms=100.0,
                preference_stability=1.1,
                preference_delay=0.5,
                preference_energy=0.5,
            )


if __name__ == "__main__":
    unittest.main()
