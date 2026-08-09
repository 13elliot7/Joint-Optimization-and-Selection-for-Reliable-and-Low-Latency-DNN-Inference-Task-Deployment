from __future__ import annotations

import dataclasses
import random
import unittest

from core.environment import Environment


class ObservationSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(20260802)
        self.environment = Environment(t_max=1, verbose=False)

    def test_snapshot_is_complete_immutable_and_has_no_future_fields(self) -> None:
        snapshot = self.environment.capture_observation_snapshot("planning")
        self.assertEqual(len(snapshot.node_online), len(self.environment.nodes))
        self.assertEqual(len(snapshot.node_available_cpu), len(self.environment.nodes))
        self.assertEqual(len(snapshot.directional_link_loads), len(self.environment.link_nodes))
        self.assertEqual(len(snapshot.physical_link_heats), len(self.environment.physical_links))
        self.assertFalse(hasattr(snapshot, "future_requests"))
        self.assertFalse(hasattr(snapshot, "remaining_time"))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            snapshot.slot = 3  # type: ignore[misc]

    def test_snapshot_versions_increase_without_changing_environment_state(self) -> None:
        first = self.environment.capture_observation_snapshot("planning")
        second = self.environment.capture_observation_snapshot("publication_revalidation")
        self.assertEqual(second.snapshot_version, first.snapshot_version + 1)
        self.assertEqual(second.environment_state_version, first.environment_state_version)

    def test_successful_admission_invalidates_previous_state_version(self) -> None:
        before = self.environment.capture_observation_snapshot("online_admission")
        dnn = self.environment.ds[0]
        assignment = [dnn.initiateNode] * len(dnn.tasks)
        self.environment.add_running_dnn(0, assignment, estimated_runtime=100.0, remaining_slots=1)
        after = self.environment.capture_observation_snapshot("online_admission")
        self.assertGreater(after.environment_state_version, before.environment_state_version)

    def test_advance_time_slot_invalidates_previous_state_version(self) -> None:
        before = self.environment.capture_observation_snapshot()
        self.environment.advance_time_slot()
        after = self.environment.capture_observation_snapshot()
        self.assertGreater(after.environment_state_version, before.environment_state_version)
        self.assertEqual(after.slot, before.slot + 1)

    def test_unknown_snapshot_purpose_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            self.environment.capture_observation_snapshot("future")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
