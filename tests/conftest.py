import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits


@pytest.fixture(scope="session", autouse=True)
def limit_threads():
    with threadpool_limits(limits=2):
        yield


@pytest.fixture(scope="session")
def prices():
    rng = np.random.default_rng(42)
    count = 1500
    returns = np.zeros(count)
    variance = np.full(count, 0.0001)
    for i in range(1, count):
        variance[i] = 0.000002 + 0.08 * returns[i - 1] ** 2 + 0.9 * variance[i - 1]
        returns[i] = np.sqrt(variance[i]) * rng.normal()
    close = 100 * np.exp(np.cumsum(returns))
    spread = np.abs(rng.normal(0.008, 0.003, count))
    return pd.DataFrame(
        {
            "open": close * np.exp(-returns / 2),
            "high": close * np.exp(np.maximum(-returns / 2, 0) + spread),
            "low": close * np.exp(np.minimum(-returns / 2, 0) - spread),
            "close": close,
            "adj_close": close,
            "volume": rng.integers(1_000_000, 5_000_000, count),
        },
        index=pd.bdate_range("2010-01-04", periods=count, name="date"),
    )
