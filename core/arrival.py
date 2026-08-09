from __future__ import annotations

import math
import random
from dataclasses import dataclass

from models import DNNProfileCatalog, InferenceRequest


@dataclass(frozen=True)
class PoissonArrivalConfig:
    """独立于算法随机数流的泊松请求生成配置。"""

    lambda_per_slot: float
    origin_nodes: tuple[int, ...]
    min_deadline_ms: float = 500.0
    max_deadline_ms: float = 1999.0
    min_preference: float = 0.7
    max_preference: float = 1.0

    def __post_init__(self) -> None:
        if self.lambda_per_slot < 0.0:
            raise ValueError("lambda_per_slot must be non-negative")
        if not self.origin_nodes:
            raise ValueError("origin_nodes must not be empty")
        if any(node < 0 for node in self.origin_nodes):
            raise ValueError("origin node indices must be non-negative")
        if self.min_deadline_ms <= 0.0 or self.max_deadline_ms < self.min_deadline_ms:
            raise ValueError("invalid deadline interval")
        if not 0.0 <= self.min_preference <= self.max_preference <= 1.0:
            raise ValueError("preference interval must be within [0, 1]")


@dataclass(frozen=True)
class ArrivalBatch:
    slot: int
    requests: tuple[InferenceRequest, ...]

    def __post_init__(self) -> None:
        if self.slot < 0:
            raise ValueError("slot must be non-negative")
        if any(request.arrival_slot != self.slot for request in self.requests):
            raise ValueError("all requests in a batch must match its slot")


@dataclass(frozen=True)
class ArrivalTrace:
    """环境持有的公平请求轨迹；算法接口不得直接持有该对象。"""

    batches: tuple[ArrivalBatch, ...]
    seed: int
    semantics_version: str = "poisson_profile_catalog_v1"

    def __post_init__(self) -> None:
        slots = [batch.slot for batch in self.batches]
        if slots != list(range(len(slots))):
            raise ValueError("arrival trace batches must cover consecutive slots from zero")

    def environment_requests_at(self, slot: int) -> tuple[InferenceRequest, ...]:
        """仅供环境事件循环在当前时隙暴露请求。"""
        if slot < 0 or slot >= len(self.batches):
            return ()
        return self.batches[slot].requests


class PoissonArrivalProcess:
    """使用私有 RNG 生成画像请求，避免算法搜索改变请求轨迹。"""

    def __init__(
        self,
        catalog: DNNProfileCatalog,
        config: PoissonArrivalConfig,
        *,
        seed: int,
    ) -> None:
        self.catalog = catalog
        self.config = config
        self.seed = seed
        self._rng = random.Random(seed)
        self._next_request_id = 0

    def _sample_poisson(self, rate: float) -> int:
        """用泊松可加性分块执行 Knuth 精确采样，避免大 rate 下下溢。"""
        if rate <= 0.0:
            return 0
        if rate > 20.0:
            chunk_count = int(math.ceil(rate / 20.0))
            chunk_rate = rate / chunk_count
            return sum(self._sample_poisson(chunk_rate) for _ in range(chunk_count))
        threshold = math.exp(-rate)
        product = 1.0
        count = 0
        while product > threshold:
            count += 1
            product *= self._rng.random()
        return count - 1

    def _sample_profile_id(self) -> str:
        return self._rng.choices(
            [profile.profile_id for profile in self.catalog.profiles],
            weights=self.catalog.sampling_weights,
            k=1,
        )[0]

    def requests_at(self, slot: int) -> tuple[InferenceRequest, ...]:
        """因果地生成一个时隙的请求批次。"""
        if slot < 0:
            raise ValueError("slot must be non-negative")
        request_count = self._sample_poisson(self.config.lambda_per_slot)
        requests = []
        for _ in range(request_count):
            request = InferenceRequest(
                request_id=self._next_request_id,
                profile_id=self._sample_profile_id(),
                arrival_slot=slot,
                initiate_node=self._rng.choice(self.config.origin_nodes),
                deadline_ms=self._rng.uniform(
                    self.config.min_deadline_ms,
                    self.config.max_deadline_ms,
                ),
                preference_stability=self._rng.uniform(
                    self.config.min_preference,
                    self.config.max_preference,
                ),
                preference_delay=self._rng.uniform(
                    self.config.min_preference,
                    self.config.max_preference,
                ),
                preference_energy=self._rng.uniform(
                    self.config.min_preference,
                    self.config.max_preference,
                ),
            )
            requests.append(request)
            self._next_request_id += 1
        return tuple(requests)

    def generate_trace(self, slot_count: int) -> ArrivalTrace:
        """为跨算法公平实验预生成只由环境持有的完整请求轨迹。"""
        if slot_count < 0:
            raise ValueError("slot_count must be non-negative")
        batches = tuple(
            ArrivalBatch(slot=slot, requests=self.requests_at(slot))
            for slot in range(slot_count)
        )
        return ArrivalTrace(batches=batches, seed=self.seed)
