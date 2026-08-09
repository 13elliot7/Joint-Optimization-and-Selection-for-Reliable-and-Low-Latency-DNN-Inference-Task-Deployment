from __future__ import annotations

import dataclasses
import random
import unittest

from core.environment import Environment, StaleEnvironmentStateError
from models import DNN, InferenceRequest, Task
from planning.online_dispatcher import OnlineDispatcher
from planning.repository import PlanRepository, build_plan_repository, make_deployment_plan
from semantics import PERIODIC_SEMANTIC_VERSIONS


class OnlineDispatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(20260807)
        self.environment = Environment(
            t_max=2,
            objective_semantics_version=PERIODIC_SEMANTIC_VERSIONS.objective,
            verbose=False,
        )
        self.user = next(
            index for index, node in enumerate(self.environment.nodes) if node.level == 1
        )
        self.edges = [
            index for index, node in enumerate(self.environment.nodes) if node.level == 2
        ]
        self.semantic_versions = PERIODIC_SEMANTIC_VERSIONS.as_pairs()
        self.origin_group = self.environment.origin_group_for_node(self.user)

    def _install_requests(self, cpu_need: int = 1) -> None:
        for index in range(2):
            self.environment.ds[index] = DNN(
                tasks=[Task(cpu_need, 10.0)],
                links=[],
                delay=100_000,
                initiateNode=self.user,
                profile_id="profile-a",
            )

    def _request(self, request_id: int) -> InferenceRequest:
        return InferenceRequest(
            request_id=request_id,
            profile_id="profile-a",
            arrival_slot=0,
            initiate_node=self.user,
            deadline_ms=100_000.0,
            preference_stability=0.4,
            preference_delay=0.3,
            preference_energy=0.3,
        )

    def _repository(self, node_index: int):
        snapshot = self.environment.capture_observation_snapshot("planning")
        candidate = self.environment.predict_candidate_state(0, [node_index], snapshot)
        plan = make_deployment_plan(
            self.environment,
            0,
            candidate,
            profile_id="profile-a",
            origin_group=self.origin_group,
            created_slot=0,
            ttl_slots=10,
            semantic_versions=self.semantic_versions,
            plan_id="cached-plan",
        )
        # 极端参考值用于证明在线结果不是直接照抄规划指标。
        plan = dataclasses.replace(
            plan,
            planning_reference_delay_ms=0.0,
            planning_reference_oss=1.0,
        )
        return build_plan_repository(
            1,
            self.environment.topology_version,
            self.semantic_versions,
            0,
            [plan],
        )

    def test_direct_plan_is_revalidated_and_committed(self) -> None:
        self._install_requests()
        repository = self._repository(self.edges[0])
        before_version = self.environment.environment_state_version
        decision = OnlineDispatcher(self.environment).dispatch(
            0, self._request(0), repository
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.path, "plan_direct")
        self.assertIsNotNone(decision.candidate)
        assert decision.candidate is not None
        self.assertGreater(decision.candidate.estimated_delay_ms, 0.0)
        self.assertGreater(decision.full_revalidation_count, 0)
        self.assertGreater(self.environment.environment_state_version, before_version)
        self.assertEqual(len(self.environment.running_dnns), 1)

    def test_offline_cached_plan_is_repaired_before_commit(self) -> None:
        self._install_requests()
        cached_node = self.edges[0]
        repository = self._repository(cached_node)
        self.environment.up[cached_node] = False
        decision = OnlineDispatcher(self.environment).dispatch(
            0, self._request(0), repository
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.path, "plan_repaired")
        self.assertNotIn(cached_node, decision.assignment or ())

    def test_same_slot_requests_cannot_overcommit_the_cached_node(self) -> None:
        target = self.edges[0]
        cpu_need = self.environment.nodes[target].max_cpu // 2 + 1
        self._install_requests(cpu_need=cpu_need)
        repository = self._repository(target)
        for index in range(self.environment.availability_node_count):
            self.environment.up[index] = index == target
        dispatcher = OnlineDispatcher(self.environment)

        first = dispatcher.dispatch(0, self._request(0), repository)
        second = dispatcher.dispatch(1, self._request(1), repository)

        self.assertTrue(first.accepted)
        self.assertFalse(second.accepted)
        self.assertGreaterEqual(self.environment.nodes[target].cpu, 0)
        self.assertEqual(len(self.environment.running_dnns), 1)
        self.assertGreater(
            second.evaluated_environment_state_version,
            first.evaluated_environment_state_version,
        )

    def test_commit_rejects_stale_environment_version(self) -> None:
        self._install_requests()
        expected = self.environment.environment_state_version
        self.environment.advance_time_slot()
        with self.assertRaises(StaleEnvironmentStateError):
            self.environment.commit_assignment(0, [self.edges[0]], expected)

    def test_empty_repository_uses_fast_fallback(self) -> None:
        self._install_requests()
        repository = PlanRepository(
            repository_version=0,
            topology_version=self.environment.topology_version,
            semantic_versions=self.semantic_versions,
            published_slot=0,
            plans=(),
        )
        decision = OnlineDispatcher(self.environment).dispatch(
            0, self._request(0), repository
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.path, "fast_fallback")
        self.assertIsNone(decision.plan_id)

    def test_stale_commit_triggers_one_current_state_revalidation(self) -> None:
        self._install_requests()
        repository = self._repository(self.edges[0])
        original_commit = self.environment.commit_assignment
        calls = 0

        def flaky_commit(dnn_index, assignment, expected_environment_state_version):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise StaleEnvironmentStateError("simulated concurrent change")
            return original_commit(
                dnn_index,
                assignment,
                expected_environment_state_version,
            )

        self.environment.commit_assignment = flaky_commit  # type: ignore[method-assign]
        decision = OnlineDispatcher(self.environment).dispatch(
            0, self._request(0), repository
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(calls, 2)
        self.assertGreaterEqual(decision.full_revalidation_count, 2)


if __name__ == "__main__":
    unittest.main()
