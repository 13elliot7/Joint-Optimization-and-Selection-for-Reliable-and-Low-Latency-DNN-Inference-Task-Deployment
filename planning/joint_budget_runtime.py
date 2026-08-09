from __future__ import annotations

import threading

from planning.control_models import JointBudgetDecision
from planning.online_dispatcher import OnlineDispatcher
from planning.periodic_planner import PeriodicPlanner


class JointBudgetRuntimeCoordinator:
    """在一个短临界区内把同一决策的 T、K、B 同时应用到运行时。"""

    def __init__(
        self,
        planner: PeriodicPlanner,
        dispatcher: OnlineDispatcher,
    ) -> None:
        if planner.environment is not dispatcher.environment:
            raise ValueError("planner and dispatcher must share one environment")
        self.planner = planner
        self.dispatcher = dispatcher
        self.current_decision: JointBudgetDecision | None = None
        self._lock = threading.Lock()

    def apply(self, decision: JointBudgetDecision) -> bool:
        """规划作业运行时拒绝切换；失败前不修改任一运行组件。"""
        with self._lock:
            if self.planner.active_job is not None:
                return False
            target_keys = {
                (target.profile_id, target.origin_group)
                for target in self.planner.targets
            }
            decision_keys = {
                (profile_id, origin_group)
                for profile_id, origin_group, _ in decision.pool_budget_by_key
            }
            if target_keys - decision_keys:
                return False
            self.planner.apply_joint_budget_decision(decision)
            self.dispatcher.apply_joint_budget_decision(decision)
            self.current_decision = decision
            return True
