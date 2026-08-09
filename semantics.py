from __future__ import annotations

from dataclasses import dataclass


LEGACY_OBJECTIVE_SEMANTICS_VERSION = "stability_fidelity_v2_return_energy"
ACTIVE_OBJECTIVE_SEMANTICS_VERSION = "stability_delay_energy_v4"


@dataclass(frozen=True)
class SemanticVersions:
    """一次运行所使用的机器可读模型语义版本集合。"""

    objective: str
    candidate_prediction: str
    observation: str
    workload: str
    planning: str
    candidate_generator: str
    availability_estimator: str
    availability_model: str
    runtime_progress: str
    routing: str
    duplex: str
    energy: str
    result_schema: str
    joint_budget_control: str
    joint_action_space: str
    pool_allocation: str
    online_anytime_revalidation: str
    action_predictor: str
    control_objective: str

    def as_pairs(self) -> tuple[tuple[str, str], ...]:
        """返回可稳定序列化和参与缓存键计算的版本对。"""
        return tuple(
            (name, value)
            for name, value in (
                ("objective_semantics_version", self.objective),
                ("candidate_prediction_version", self.candidate_prediction),
                ("observation_semantics_version", self.observation),
                ("workload_semantics_version", self.workload),
                ("planning_semantics_version", self.planning),
                ("candidate_generator_version", self.candidate_generator),
                ("availability_estimator_version", self.availability_estimator),
                ("availability_model_version", self.availability_model),
                ("runtime_progress_mode", self.runtime_progress),
                ("routing_mode", self.routing),
                ("duplex_mode", self.duplex),
                ("energy_model_version", self.energy),
                ("result_schema_version", self.result_schema),
                ("joint_budget_control_version", self.joint_budget_control),
                ("joint_action_space_version", self.joint_action_space),
                ("pool_allocation_version", self.pool_allocation),
                (
                    "online_anytime_revalidation_version",
                    self.online_anytime_revalidation,
                ),
                ("action_predictor_version", self.action_predictor),
                ("control_objective_version", self.control_objective),
            )
        )


LEGACY_SEMANTIC_VERSIONS = SemanticVersions(
    objective=LEGACY_OBJECTIVE_SEMANTICS_VERSION,
    candidate_prediction="legacy_one_step_v2",
    observation="implicit_environment_v1",
    workload="deterministic_generated_dnn_v1",
    planning="per_request_search_v1",
    candidate_generator="legacy_per_request_v1",
    availability_estimator="rtbl_history_v1",
    availability_model="partial_node_up_down_v1",
    runtime_progress="fixed_service_time_v1",
    routing="fixed_path_v1",
    duplex="full_duplex_shared_heat_v1",
    energy="deployment_return_energy_v1",
    result_schema="semantic_names_v4_global",
    joint_budget_control="fixed_legacy_v1",
    joint_action_space="fixed_legacy_v1",
    pool_allocation="global_fixed_pool_v1",
    online_anytime_revalidation="fixed_top_k_v1",
    action_predictor="none_legacy_v1",
    control_objective="none_legacy_v1",
)


PERIODIC_SEMANTIC_VERSIONS = SemanticVersions(
    objective=ACTIVE_OBJECTIVE_SEMANTICS_VERSION,
    candidate_prediction="unified_fixed_point_v3",
    observation="explicit_snapshot_v1",
    workload="poisson_profile_catalog_v1",
    planning="periodic_repository_v1",
    candidate_generator="customized_periodic_pool_v1",
    availability_estimator="empirical_window_beta_v1",
    availability_model="slot_up_down_v1",
    runtime_progress="fixed_service_time_v1",
    routing="fixed_path_v1",
    duplex="full_duplex_shared_heat_v1",
    energy="deployment_return_energy_v1",
    result_schema="periodic_three_objective_v3",
    joint_budget_control="discrete_mpc_v1",
    joint_action_space="grid_tkb_v1",
    pool_allocation="arrival_risk_diversity_v1",
    online_anytime_revalidation="budgeted_fixed_point_v1",
    action_predictor="regularized_surrogate_v1",
    control_objective="goodput_cost_v1",
)
