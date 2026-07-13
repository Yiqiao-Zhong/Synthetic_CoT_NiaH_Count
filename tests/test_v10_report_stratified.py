import pandas as pd
import pytest

from synthetic_counting_v10.report_stratified import (
    _linear_summary,
    count_bin_from_value,
)


@pytest.mark.parametrize(
    ("count", "expected"),
    [(1, "1-10"), (10, "1-10"), (11, "11-20"), (20, "11-20"), (21, "21-30"), (30, "21-30")],
)
def test_count_bin_from_value(count, expected):
    assert count_bin_from_value(count) == expected


def test_linear_summary_recovers_transport_slope():
    frame = pd.DataFrame(
        {
            "count_bin": ["1-10"] * 4,
            "receiver_count": [1, 2, 3, 4],
            "predicted_count": [3, 5, 7, 9],
        }
    )

    result = _linear_summary(
        frame,
        ["count_bin"],
        x_column="receiver_count",
        y_column="predicted_count",
    ).iloc[0]

    assert result["slope"] == pytest.approx(2.0)
    assert result["intercept"] == pytest.approx(1.0)
    assert result["r2"] == pytest.approx(1.0)
