def choose_command(domain, position, heading, speed_mps, nominal_action,
                   target_speed, others=(), other_lifetimes=(), goal=None,
                   goal_radius=5.0, separation_km=10.26):
    """Select a nearby turn/speed command, with explicit infeasibility diagnostics.

    target_speed maps normalized speed commands to predicted true airspeed. Static
    feasibility is prioritized if no candidate satisfies both kinds of constraint.
    """
    nominal = np.clip(np.asarray(nominal_action, dtype=float), -1, 1)
    def assess(actions):
        paths = predict_commands(position, heading, speed_mps, actions[:, 0] * 45,
                                 target_speed(actions[:, 1]))
        life = capture_times(paths, goal, goal_radius)
        static_paths = stop_at_goal(paths, goal, goal_radius) if goal is not None else paths
        trajectories = linestrings(static_paths)
        static = np.asarray(covers(domain, trajectories), dtype=bool)
        traffic, risk, minimum = traffic_risk(paths, life, others, other_lifetimes, separation_km)
        return paths, life, trajectories, static, traffic, risk, minimum
    initial = assess(nominal[None])
    nominal_static, nominal_traffic = bool(initial[3][0]), bool(initial[4][0])
    if nominal_static and nominal_traffic:
        chosen, results, index = nominal, initial, 0
    else:
        turns = np.unique(np.r_[nominal[0], np.linspace(-1, 1, 19)])
        speeds = np.unique(np.r_[nominal[1], -3, -1, 0, 1])
        actions = np.array(np.meshgrid(turns, speeds, indexing='ij')).reshape(2, -1).T
        results = assess(actions)
        paths, life, trajectories, static, traffic, risk, minimum = results
        joint = static & traffic
        motion = np.abs(actions[:, 0] - nominal[0]) + .35 * np.abs(actions[:, 1] - nominal[1])
        if joint.any():
            candidates = np.flatnonzero(joint)
            index = candidates[np.lexsort((-actions[candidates, 0], motion[candidates]))[0]]
        elif static.any():
            candidates = np.flatnonzero(static)
            index = candidates[np.lexsort((-actions[candidates, 0], motion[candidates], risk[candidates]))[0]]
        else:
            if domain.is_empty:
                static_cost = np.zeros(len(actions))
            else:
                static_cost = length(difference(trajectories, domain)) + 4 * distance(get_point(trajectories, -1), domain)
            index = np.lexsort((-actions[:, 0], motion, risk, static_cost))[0]
        chosen = actions[index]
    paths, life, _, static, traffic, risk, minimum = results
    diagnostics = dict(nominal_static_violation=not nominal_static, nominal_traffic_conflict=not nominal_traffic,
                       no_static_feasible_candidate=not bool(static.any()),
                       no_jointly_feasible_candidate=not bool((static & traffic).any()),
                       predicted_minimum_separation_km=float(minimum[index]),
                       predicted_risk_seconds=float(risk[index]))
    return chosen, paths[index], float(life[index]), diagnostics
