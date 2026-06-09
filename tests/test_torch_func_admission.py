import pytest

import vptune as vp
import vptune.ext as vpx


def torch_func_settings(**overrides: object) -> dict[str, object]:
    return {
        "contains_autograd_call": False,
        "contains_backward_call": False,
        "uses_out_variant": False,
        "uses_data_dependent_control_flow": False,
        "uses_item": False,
        "has_dynamic_shape_output": False,
        "vectorization.randomness": "error",
        "requires_forward_ad": False,
        "forward_ad_supported": False,
        **overrides,
    }


@pytest.mark.parametrize(
    "randomness",
    [
        "error",
        "same",
        "different",
    ],
)
def test_torch_func_admission_accepts_declared_vmap_randomness(
    randomness: str,
) -> None:
    assert (
        vpx.admit_torch_func(
            torch_func_settings(**{"vectorization.randomness": randomness})
        )
        is None
    )


@pytest.mark.parametrize(
    "field",
    [
        "uses_item",
        "has_dynamic_shape_output",
        "uses_data_dependent_control_flow",
        "contains_autograd_call",
        "contains_backward_call",
        "uses_out_variant",
    ],
)
def test_torch_func_admission_rejects_transform_limitations(field: str) -> None:
    with pytest.raises(vp.AdmissionError, match=field):
        vpx.admit_torch_func(torch_func_settings(**{field: True}))


def test_torch_func_admission_rejects_forward_ad_coverage_failure() -> None:
    with pytest.raises(vp.AdmissionError, match="forward AD"):
        vpx.admit_torch_func(
            torch_func_settings(
                requires_forward_ad=True,
                forward_ad_supported=False,
            )
        )


def test_torch_func_admission_rejects_invalid_vmap_randomness() -> None:
    with pytest.raises(vp.AdmissionError, match=r"vectorization\.randomness"):
        vpx.admit_torch_func(
            torch_func_settings(**{"vectorization.randomness": "random"})
        )


def test_torch_func_admission_requires_boolean_flags() -> None:
    with pytest.raises(vp.AdmissionError, match="uses_item must be a bool"):
        vpx.admit_torch_func(torch_func_settings(uses_item="false"))
