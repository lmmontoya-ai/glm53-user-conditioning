"""Strict records for the GLM-5.3 user/evaluation project."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SHA_PATTERN = r"^[0-9a-f]{40,64}$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GitLock(StrictModel):
    repo: str = Field(min_length=3)
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")


class ProjectLock(StrictModel):
    repo: str = Field(min_length=3)
    starting_commit: str = Field(pattern=r"^[0-9a-f]{40}$")


class ModelLock(StrictModel):
    repo: str = Field(min_length=3)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    safetensor_shards: int = Field(gt=0)
    safetensor_bytes: int = Field(gt=0)


class SourceLocks(StrictModel):
    schema_version: Literal["glm53_source_locks_v1"]
    project_repo: ProjectLock
    transluce: GitLock
    hua: GitLock
    transformers: GitLock
    model: ModelLock


class PersonaPair(StrictModel):
    schema_version: Literal["glm53_persona_pair_v1"] = "glm53_persona_pair_v1"
    twin_index: int = Field(ge=0)
    famous_key: str = Field(min_length=1)
    unknown_ai_key: str = Field(min_length=1)
    genpop_key: str = Field(min_length=1)
    famous_name: str = Field(min_length=1)
    unknown_name: str = Field(min_length=1)
    organization: str = Field(min_length=1)
    prior_glm52_name_effect_pp: float
    selection_role: Literal["enriched_target", "prospective_generality"]
    primary_intervention: bool


class ControlPersona(StrictModel):
    key: str = Field(min_length=1)
    name: str = Field(min_length=1)
    group: Literal["famous_nonai", "genpop"]


class BehaviorScheduleRow(StrictModel):
    schema_version: Literal["glm53_behavior_schedule_v1"] = "glm53_behavior_schedule_v1"
    sample_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    persona_key: str = Field(min_length=1)
    pair_index: int | None = Field(default=None, ge=0)
    condition: str = Field(min_length=1)
    phase: str = Field(min_length=1)
    reasoning_effort: Literal["low", "high", "max"]
    generation_seed: int
    prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider: str = Field(min_length=1)
    model_id: str = Field(min_length=1)


class BehaviorPromptRecord(StrictModel):
    schema_version: Literal["glm53_behavior_prompt_v1"] = "glm53_behavior_prompt_v1"
    sample_id: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)
    main_prompt: str = Field(min_length=1)
    followup_prompt: str = Field(min_length=1)
    prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class BehaviorResultRow(StrictModel):
    schema_version: Literal["glm53_behavior_result_v1"] = "glm53_behavior_result_v1"
    run_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    persona_key: str = Field(min_length=1)
    condition: str = Field(min_length=1)
    binary_answer: Literal["yes", "no"] | None
    confidence_p: float | None = Field(default=None, ge=0.0, le=100.0)
    refused_confidence: bool
    judge_model: str = Field(min_length=1)
    judge_text: str
    subject_response_main: str
    subject_response_followup: str
    realized_reasoning_tokens: int | None = Field(default=None, ge=0)
    parse_valid: bool
    provider_metadata: dict[str, Any]

    @model_validator(mode="after")
    def validate_parse_state(self) -> "BehaviorResultRow":
        if self.parse_valid and (self.binary_answer is None or self.confidence_p is None):
            raise ValueError("valid behavior rows require binary answer and confidence")
        if self.refused_confidence and self.confidence_p is not None:
            raise ValueError("a refused confidence response cannot contain a confidence score")
        return self


class LocalSubjectResult(StrictModel):
    schema_version: Literal["glm53_local_subject_result_v1"] = (
        "glm53_local_subject_result_v1"
    )
    run_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    persona_key: str = Field(min_length=1)
    condition: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    main_text: str
    followup_text: str
    main_prompt_tokens: int = Field(gt=0)
    main_generated_tokens: int = Field(ge=0)
    followup_prompt_tokens: int = Field(gt=0)
    followup_generated_tokens: int = Field(ge=0)
    main_output_token_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    followup_output_token_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_seed: int
    generation_config: dict[str, Any]
    latency_seconds: dict[str, float]
    completed_at: str


class SelfHostedSubjectResult(StrictModel):
    schema_version: Literal["glm53_self_hosted_subject_result_v1"] = (
        "glm53_self_hosted_subject_result_v1"
    )
    run_id: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    persona_key: str = Field(min_length=1)
    condition: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    serving_runtime_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    serving_engine: Literal["vllm_openai"]
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    main_text: str
    followup_text: str
    main_usage: dict[str, Any]
    followup_usage: dict[str, Any]
    main_response_id: str
    followup_response_id: str
    main_finish_reason: str | None
    followup_finish_reason: str | None
    main_response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    followup_response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_seed: int
    generation_config: dict[str, Any]
    latency_seconds: dict[str, float]
    completed_at: str


class ActivationRow(StrictModel):
    schema_version: Literal["glm53_activation_row_v1"] = "glm53_activation_row_v1"
    run_id: str
    sample_id: str
    concept: str
    class_label: Literal[0, 1]
    group_id: str
    organization: str
    view: str
    layer_index: int = Field(ge=0)
    feature_path: str
    feature_row: int = Field(ge=0)
    hidden_size: int = Field(gt=0)
    hc_mult: int = Field(gt=0)
    prompt_token_count: int = Field(gt=0)
    prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class DirectionArtifact(StrictModel):
    schema_version: Literal["glm53_direction_v1"] = "glm53_direction_v1"
    direction_id: str
    concept: Literal["eval_deploy", "recognized_name", "affiliation"]
    sign: str
    layer_index: int = Field(ge=0)
    view: str
    construction: Literal["paired_mean_difference"]
    train_split_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    vector_path: str
    vector_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    norm: float = Field(gt=0.0)
    natural_projection_gap: float
    frozen_at: str


class GateDecision(StrictModel):
    schema_version: Literal["glm53_user_eval_decision_v1"] = "glm53_user_eval_decision_v1"
    project_id: Literal["glm53_user_eval_shared_direction_v1"]
    gate: str
    run_id: str
    prereg_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    inputs: tuple[str, ...]
    estimates: dict[str, Any]
    checks: dict[str, bool]
    passed: bool
    decision: str
    reason: str
    created_at: str
