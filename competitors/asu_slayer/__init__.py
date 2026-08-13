"""Net-worth-exact challenger from github.com/emirkaanozdemr/monopoly (ASU_SLAYER)."""

from .policy import DEFAULT_CONFIG, SLAYER_V1, SlayerConfig, SlayerV1
from .scoring import (
    acquisition_gain,
    deed_worth,
    disposal_loss,
    equity,
    improvement_gain,
    liquidation_options,
    net_worth,
    strength,
)


__all__ = [
    "DEFAULT_CONFIG",
    "SLAYER_V1",
    "SlayerConfig",
    "SlayerV1",
    "acquisition_gain",
    "deed_worth",
    "disposal_loss",
    "equity",
    "improvement_gain",
    "liquidation_options",
    "net_worth",
    "strength",
]
