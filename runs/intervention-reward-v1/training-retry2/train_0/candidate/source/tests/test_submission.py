import ast
from pathlib import Path
import subprocess

import pytest

from atc.submission import _validate_configuration


def test_original_harness_changes_are_limited_to_the_two_allowed_hooks():
    root = Path(__file__).resolve().parents[1]
    original = subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}",
                                        "-C", str(root), "show", "HEAD:scripts/evaluate_competition.py"],
                                       text=True, encoding="utf-8")
    current = (root / "scripts/evaluate_competition.py").read_text(encoding="utf-8")
    def scoring_tree(source):
        tree = ast.parse(source)
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.FunctionDef) and node.name in {"make_env", "load_policy"})]
        return ast.dump(tree, include_attributes=False)
    assert scoring_tree(original) == scoring_tree(current)


def test_submission_rejects_wrong_track_and_stale_route_mapping():
    with pytest.raises(ValueError, match="track"):
        _validate_configuration("sa", {"env": "ma", "algorithm": "sac", "recipe": "public_weights"})
    with pytest.raises(ValueError, match="route revision"):
        _validate_configuration("ma", {"env": "ma", "algorithm": "sac", "recipe": "route_guided_inset", "route_revision": 1})


def test_direct_projection_configuration_requires_the_verified_parameters():
    from atc.projection import PROJECTION_REVISION, HORIZON_SECONDS, CLEARANCE_KM
    configuration = {"env": "ma", "algorithm": "sac", "recipe": "public_weights", "guard_static": True,
                     "static_projection_revision": PROJECTION_REVISION,
                     "static_projection_horizon_seconds": HORIZON_SECONDS,
                     "static_projection_clearance_km": CLEARANCE_KM}
    _validate_configuration("ma", configuration)
    configuration["static_projection_horizon_seconds"] = HORIZON_SECONDS / 2
    with pytest.raises(ValueError, match="projection settings"):
        _validate_configuration("ma", configuration)
