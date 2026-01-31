from __future__ import annotations

from typing import Dict, List, Tuple

from ortools.sat.python import cp_model


def build_rank_model(
    weeks: List[Dict],
    weights_scaled: Dict[str, int],
) -> Dict:
    season = weeks[0]["season"]
    model = cp_model.CpModel()

    rF_vars: Dict[Tuple[int, int], cp_model.IntVar] = {}
    x_vars: Dict[Tuple[int, int, int], cp_model.IntVar] = {}
    week_terms: Dict[int, Dict[str, List[cp_model.IntVar]]] = {}
    slack_by_week_contestant: Dict[Tuple[int, int], List[cp_model.IntVar]] = {}

    week_map = {w["week"]: w for w in weeks}
    week_numbers = sorted(week_map.keys())

    for week_num in week_numbers:
        week = week_map[week_num]
        contestants = week["contestants"]
        n_w = len(contestants)
        week_terms[week_num] = {"jterm": [], "smooth": [], "slack": []}

        for i in contestants:
            rF_vars[(week_num, i)] = model.NewIntVar(
                1, n_w, f"rF_s{season}_w{week_num}_i{i}"
            )

            x_terms = []
            for k in range(1, n_w + 1):
                x_var = model.NewBoolVar(
                    f"x_s{season}_w{week_num}_i{i}_k{k}"
                )
                x_vars[(week_num, i, k)] = x_var
                x_terms.append(k * x_var)

            model.Add(sum(x_vars[(week_num, i, k)] for k in range(1, n_w + 1)) == 1)
            model.Add(rF_vars[(week_num, i)] == sum(x_terms))

        for k in range(1, n_w + 1):
            model.Add(
                sum(x_vars[(week_num, i, k)] for i in contestants) == 1
            )

        for i in contestants:
            rJ = int(week["judge_rank"][i])
            d_var = model.NewIntVar(
                -n_w, n_w, f"d_j_s{season}_w{week_num}_i{i}"
            )
            d_sq = model.NewIntVar(
                0, n_w * n_w, f"d_sq_j_s{season}_w{week_num}_i{i}"
            )
            model.Add(d_var == rF_vars[(week_num, i)] - rJ)
            model.AddMultiplicationEquality(d_sq, [d_var, d_var])
            week_terms[week_num]["jterm"].append(d_sq)

        eliminated = week["eliminated_ids"]
        if eliminated:
            for e in eliminated:
                rJ_e = int(week["judge_rank"][e])
                for j in contestants:
                    if j == e:
                        continue
                    rJ_j = int(week["judge_rank"][j])
                    delta = model.NewIntVar(
                        0, 2 * n_w, f"delta_s{season}_w{week_num}_e{e}_j{j}"
                    )
                    model.Add(
                        rF_vars[(week_num, e)] + rJ_e + delta
                        >= rF_vars[(week_num, j)] + rJ_j + 1
                    )
                    week_terms[week_num]["slack"].append(delta)
                    slack_by_week_contestant.setdefault((week_num, j), []).append(
                        delta
                    )

    for week_num in week_numbers:
        prev_week = week_num - 1
        if prev_week not in week_map:
            continue
        week = week_map[week_num]
        prev = week_map[prev_week]
        shared = set(week["contestants"]).intersection(prev["contestants"])
        if not shared:
            continue
        max_n = max(len(week["contestants"]), len(prev["contestants"]))
        for i in shared:
            d_var = model.NewIntVar(
                -max_n, max_n, f"d_sm_s{season}_w{week_num}_i{i}"
            )
            d_sq = model.NewIntVar(
                0, max_n * max_n, f"d_sq_sm_s{season}_w{week_num}_i{i}"
            )
            model.Add(d_var == rF_vars[(week_num, i)] - rF_vars[(prev_week, i)])
            model.AddMultiplicationEquality(d_sq, [d_var, d_var])
            week_terms[week_num]["smooth"].append(d_sq)

    objective_terms = []
    for week_num in week_numbers:
        for term in week_terms[week_num]["jterm"]:
            objective_terms.append(weights_scaled["alpha"] * term)
        for term in week_terms[week_num]["smooth"]:
            objective_terms.append(weights_scaled["beta"] * term)
        for term in week_terms[week_num]["slack"]:
            objective_terms.append(weights_scaled["gamma"] * term)

    objective_expr = sum(objective_terms) if objective_terms else None

    return {
        "season": season,
        "model": model,
        "rF_vars": rF_vars,
        "week_terms": week_terms,
        "slack_by_week_contestant": slack_by_week_contestant,
        "week_map": week_map,
        "week_numbers": week_numbers,
        "objective_terms": objective_terms,
        "objective_expr": objective_expr,
    }


def solve_season(
    weeks: List[Dict],
    weights_scaled: Dict[str, int],
    weight_scale: int,
    time_limit: float,
    num_workers: int,
) -> Dict:
    if not weeks:
        return {
            "status": "NO_WEEKS",
            "objective_scaled": None,
            "objective": None,
            "rF": {},
            "week_terms": {},
            "slack_by_week_contestant": {},
        }

    build = build_rank_model(weeks, weights_scaled)
    model = build["model"]
    objective_expr = build["objective_expr"]
    if objective_expr is not None:
        model.Minimize(objective_expr)

    rF_vars = build["rF_vars"]
    week_terms = build["week_terms"]
    slack_by_week_contestant = build["slack_by_week_contestant"]
    week_numbers = build["week_numbers"]

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    if num_workers:
        solver.parameters.num_search_workers = int(num_workers)

    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    if status not in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    ):
        return {
            "status": status_name,
            "objective_scaled": None,
            "objective": None,
            "rF": {},
            "week_terms": {},
            "slack_by_week_contestant": {},
        }

    rF = {
        key: int(solver.Value(var))
        for key, var in rF_vars.items()
    }

    week_term_values = {}
    for week_num in week_numbers:
        jterm_val = sum(int(solver.Value(v)) for v in week_terms[week_num]["jterm"])
        smooth_val = sum(
            int(solver.Value(v)) for v in week_terms[week_num]["smooth"]
        )
        slack_val = sum(int(solver.Value(v)) for v in week_terms[week_num]["slack"])
        week_term_values[week_num] = {
            "jterm": jterm_val,
            "smooth": smooth_val,
            "slack": slack_val,
        }

    slack_values = {
        key: sum(int(solver.Value(v)) for v in vars_list)
        for key, vars_list in slack_by_week_contestant.items()
    }

    objective_scaled = int(solver.ObjectiveValue())
    objective = (
        float(objective_scaled) / float(weight_scale) if weight_scale else None
    )

    return {
        "status": status_name,
        "objective_scaled": objective_scaled,
        "objective": objective,
        "rF": rF,
        "week_terms": week_term_values,
        "slack_by_week_contestant": slack_values,
    }
