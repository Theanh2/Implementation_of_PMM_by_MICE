from __future__ import annotations

"""Utility helpers for the *zh* sub-package.

Currently this module offers a single public function, :func:`get_imputer_func`,
which maps a string identifying an imputation method to the concrete callable
that implements that method.

The mapping deliberately excludes the *Random Forest* (``rf``) strategy for the
moment because it is not yet implemented in this code base.
"""

from .constants import ImputationMethod

# Import concrete imputation back-ends
from imputation.PMM import pmm
from imputation.midas import midas
from imputation.cart import mice_impute_cart
from imputation.sample import mice_impute_sample

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

# Map of method-name -> imputer function
_IMPUTER_MAP = {
    ImputationMethod.PMM.value: pmm,
    ImputationMethod.MIDAS.value: midas,
    ImputationMethod.CART.value: mice_impute_cart,
    ImputationMethod.SAMPLE.value: mice_impute_sample,
}


def get_imputer_func(method_name: str):
    """Return the imputer callable for *method_name*.

    Parameters
    ----------
    method_name : str
        Name of the imputation method. Must be one of the values defined in
        :class:`imputation.zh.constants.ImputationMethod` (except 'rf', which
        is not yet available).

    Returns
    -------
    Callable
        The function implementing the requested imputation strategy.

    Raises
    ------
    ValueError
        If *method_name* is unknown or not yet implemented.
    """
    if method_name not in _IMPUTER_MAP:
        raise ValueError(
            "Unsupported or unimplemented imputation method: "
            f"'{method_name}'. Supported methods are: {list(_IMPUTER_MAP.keys())}"
        )

    return _IMPUTER_MAP[method_name] 