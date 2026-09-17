"""Select existing action filters without changing their controller behavior."""


def guard_settings(config):
    """The legacy full filter includes static areas and aircraft traffic."""
    traffic = bool(config.get("filter", False))
    static = traffic or bool(config.get("static_filter", False))
    return static, traffic


def verify_guard_selection(environment, config):
    """Record actual simulator mixins, rejecting silently ignored support flags."""
    from atc.projection import StaticActionProjection
    from atc.traffic_projection import TrafficActionProjection
    world = environment.unwrapped
    actual = (isinstance(world, StaticActionProjection),
              isinstance(world, TrafficActionProjection))
    if actual != guard_settings(config):
        raise ValueError("Simulator action filters differ from the requested support")
    return {"static_area_filter": actual[0], "traffic_conflict_filter": actual[1]}
