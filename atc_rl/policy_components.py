"""Closed-loop heading/speed swaps between fixed initial and trained policies."""
import numpy as np


def combine_policy_components(initial, trained, heading_source, speed_source):
    """Select complete commands before the unchanged environment action filters."""
    first=np.asarray(initial,dtype=np.float32)
    final=np.asarray(trained,dtype=np.float32)
    if first.shape!=final.shape or first.ndim not in (1,2) or first.shape[-1]!=2:
        raise ValueError("Expected aligned heading/speed command pairs")
    if not all(np.isfinite(x).all() and np.all(np.abs(x)<=1) for x in (first,final)):
        raise ValueError("Policy commands must be finite and normalized")
    if heading_source not in ("initial","trained") or speed_source not in ("initial","trained"):
        raise ValueError("Each component must name its recorded policy source")
    sources={"initial":first,"trained":final}
    return np.stack((sources[heading_source][...,0],sources[speed_source][...,1]),axis=-1)
