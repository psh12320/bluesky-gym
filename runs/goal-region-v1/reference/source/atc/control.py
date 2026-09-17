import numpy as np


def goal_relative_command(action, heading, goal_bearing, turn_limit=45.0, offset_limit=90.0):
    """Map a learned goal-relative heading offset to the allowed turn command."""
    desired = goal_bearing + float(np.clip(action, -1.0, 1.0)) * offset_limit
    delta = (desired - heading + 180.0) % 360.0 - 180.0
    return float(np.clip(delta / turn_limit, -1.0, 1.0))


class GoalRelativeControl:
    """Action hook shared by the official single- and multi-aircraft environments."""

    def _get_action(self, ac_id, action):
        import bluesky as bs
        from core.tools import kwikqdrdist

        idx = bs.traf.id2idx(ac_id)
        goal = self._goal[ac_id] if hasattr(self, "_goal") else (self.goal_lat, self.goal_lon)
        bearing, _ = kwikqdrdist(bs.traf.lat[idx], bs.traf.lon[idx], *goal)
        heading_action = goal_relative_command(action[0], bs.traf.hdg[idx], bearing, self.d_heading)
        self.heading_action.execute(ac_id, heading_action)
        self.speed_action.execute(ac_id, action[1])
