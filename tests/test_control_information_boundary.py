from __future__ import annotations

import random
import unittest

from core.environment import Environment
from planning.control_state import (
    CompletedDispatchRecord,
    JointBudgetOutcomeBuilder,
    PlanningControlSnapshotBuilder,
)
from planning.control_models import JointBudgetAction, JointBudgetDecision, JointActionPrediction
from planning.repository import PlanRepository
from semantics import PERIODIC_SEMANTIC_VERSIONS


class ControlInformationBoundaryTests(unittest.TestCase):
    def test_builder_ignores_records_not_yet_causally_available(self) -> None:
        random.seed(20260810)
        environment = Environment(
            t_max=1,
            objective_semantics_version=PERIODIC_SEMANTIC_VERSIONS.objective,
            verbose=False,
        )
        for _ in range(5):
            environment.advance_time_slot()
        observation = environment.capture_observation_snapshot("planning")
        repository = PlanRepository(
            0,
            environment.topology_version,
            PERIODIC_SEMANTIC_VERSIONS.as_pairs(),
            0,
            (),
        )
        visible = CompletedDispatchRecord(
            slot=3,
            available_after_slot=4,
            profile_id="p",
            origin_group=1,
            path="plan_direct",
            online_runtime_ms=2.0,
            full_revalidation_count=1,
            feasible_candidate_count=1,
            rejected=False,
        )
        future = CompletedDispatchRecord(
            slot=4,
            available_after_slot=6,
            profile_id="future",
            origin_group=2,
            path="rejected",
            online_runtime_ms=50.0,
            full_revalidation_count=0,
            feasible_candidate_count=0,
            rejected=True,
        )
        snapshot = PlanningControlSnapshotBuilder(environment.slot_length).build(
            control_epoch=1,
            observation=observation,
            repository=repository,
            window_start_slot=0,
            dispatch_records=(visible, future),
            planning_records=(),
        )
        self.assertEqual(snapshot.direct_hit_rate, 1.0)
        self.assertEqual(snapshot.rejection_rate, 0.0)
        self.assertEqual(snapshot.arrival_rate_by_key[0][:2], ("p", 1))
        self.assertFalse(hasattr(snapshot, "future_requests"))
        self.assertFalse(hasattr(snapshot, "next_failure_slot"))

    def test_goodput_outcome_is_linked_to_decision_id(self) -> None:
        action = JointBudgetAction(100, 2, 10.0)
        prediction = JointActionPrediction(
            1.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 10.0, 5.0, 0.8
        )
        decision = JointBudgetDecision(
            "decision-1",
            1,
            0,
            "fixed_joint_v1",
            action,
            (("p", 1, 2),),
            1.0,
            prediction,
            False,
            None,
        )
        record = CompletedDispatchRecord(
            slot=1,
            available_after_slot=1,
            profile_id="p",
            origin_group=1,
            path="plan_direct",
            online_runtime_ms=1.0,
            full_revalidation_count=1,
            feasible_candidate_count=1,
            rejected=False,
            runtime_failed=False,
            deadline_violated=False,
            request_utility=0.8,
            outcome_available_after_slot=1,
        )
        outcome = JointBudgetOutcomeBuilder(100.0).build(
            decision,
            observation_end_slot=1,
            dispatch_records=(record,),
            planning_records=(),
        )
        self.assertEqual(outcome.decision_id, decision.decision_id)
        self.assertGreater(outcome.actual_goodput_utility, 0.0)


if __name__ == "__main__":
    unittest.main()
