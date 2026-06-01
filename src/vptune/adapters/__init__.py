"""Optional adapter helpers."""

from vptune.adapters.curvlinops import (
    CurvLinOpsAdmitter,
    CurvLinOpsFisherMCSemantics,
    curvlinops_axis,
    curvlinops_operation_factory,
    curvlinops_operator_axis,
    curvlinops_runtime_config,
)
from vptune.adapters.distributed import (
    DistributedAdmissionPolicy,
    RankSelectedSettings,
    RankStatus,
    admit_distributed_candidate,
    distributed_identity,
    distributed_record,
    distributed_sharding_axis,
    reduce_rank_statuses,
    require_rank_selected_settings_agree,
)
from vptune.adapters.pilot import (
    PILOT_ACCEPTANCE_FAMILIES,
    PilotReadiness,
)
from vptune.adapters.pilot import (
    acceptance_family_names as pilot_acceptance_family_names,
)
from vptune.adapters.pilot import (
    acceptance_readiness as pilot_acceptance_readiness,
)
from vptune.adapters.pilot import (
    lower as pilot_lower,
)
from vptune.adapters.pilot import (
    readiness as pilot_readiness,
)
from vptune.adapters.pilot import (
    require_acceptance_families as pilot_require_acceptance_families,
)
from vptune.adapters.pilot import (
    selected_settings as pilot_selected_settings,
)
from vptune.adapters.pilot import (
    validators as pilot_validators,
)
from vptune.adapters.transformers import (
    TransformersAttentionPolicy,
    TransformersModelIdentity,
    admit_transformers_attention,
    admit_transformers_cache,
    check_patched_attention_reference,
    check_patched_attention_vjp_reference,
    transformers_attention_axis,
    transformers_cache_axis,
    transformers_model_identity,
)

__all__ = [
    "PILOT_ACCEPTANCE_FAMILIES",
    "CurvLinOpsAdmitter",
    "CurvLinOpsFisherMCSemantics",
    "DistributedAdmissionPolicy",
    "PilotReadiness",
    "RankSelectedSettings",
    "RankStatus",
    "TransformersAttentionPolicy",
    "TransformersModelIdentity",
    "admit_distributed_candidate",
    "admit_transformers_attention",
    "admit_transformers_cache",
    "check_patched_attention_reference",
    "check_patched_attention_vjp_reference",
    "curvlinops_axis",
    "curvlinops_operation_factory",
    "curvlinops_operator_axis",
    "curvlinops_runtime_config",
    "distributed_identity",
    "distributed_record",
    "distributed_sharding_axis",
    "pilot_acceptance_family_names",
    "pilot_acceptance_readiness",
    "pilot_lower",
    "pilot_readiness",
    "pilot_require_acceptance_families",
    "pilot_selected_settings",
    "pilot_validators",
    "reduce_rank_statuses",
    "require_rank_selected_settings_agree",
    "transformers_attention_axis",
    "transformers_cache_axis",
    "transformers_model_identity",
]
