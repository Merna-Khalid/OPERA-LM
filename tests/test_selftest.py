# Re-exports opera_lm.selftest's test_* functions so pytest collects each
# arm as its own test item: a failure in one doesn't stop the others from
# running, and a single arm can be run in isolation, e.g.
#   pytest tests/test_selftest.py::test_muon_optimizer
# `python -m opera_lm.selftest` (opera_lm.selftest.selftest) runs the same
# arms sequentially instead, for a quick CLI check.
from opera_lm.selftest import *  # noqa: F401,F403
