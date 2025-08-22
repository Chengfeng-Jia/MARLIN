# unified_simulation.py
import os, time, glob, math, json, random, csv
import numpy as np
import holoocean
import gtsam
from gtsam import symbol

# ------------ 全局常量 ------------
OUT_DIR   = "uuv_simulation_output"; os.makedirs(OUT_DIR, exist_ok=True)

LOSSY_UUV   = "auv0"
LOSS_PROB   = 0.002

ASCEND_RATE = 1.0
DIVE_RATE   = 2.0

TORPEDO_KP  = 0.02
TORPEDO_MAX = 0.2

LEADER       = np.array([5.0, 5.0, 0.0])
TORPEDO_GOAL = np.array([50.0, 50.0, -15.0])

SIGMA_NAV   = 0.10
SIGMA_LOST0 = 0.2
GROWTH_RATE = 0.02
DECAY       = 0.20

# ------------ GTSAM 计算函数 ------------
def bearing_range_estimate(leader_xy, rng, brg):
    ku, kl = symbol("x", 0), symbol("l", 0)
    graph  = gtsam.NonlinearFactorGraph()
    vals   = gtsam.Values()

    graph.add(gtsam.PriorFactorPoint2(
        kl, gtsam.Point2(*leader_xy),
        gtsam.noiseModel.Isotropic.Sigma(2, 1e-6)
    ))

    br_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([np.deg2rad(10), 2.0])
    )
    graph.add(gtsam.BearingRangeFactor2D(
        ku, kl, gtsam.Rot2(brg), rng, br_noise))

    vals.insert(ku, gtsam.Pose2(0, 0, 0))
    vals.insert(kl, gtsam.Point2(*leader_xy))

    pose = gtsam.LevenbergMarquardtOptimizer(graph, vals) \
                .optimize() \
                .atPose2(ku)
    return np.array([pose.x(), pose.y(), -15.0])

# ------------ 数据预处理 ------------
def load_and_preprocess(path, scale=2.0, depth=-15):
    d = json.load(open(path))
    addz = lambda ls: [[x*scale, y*scale, depth] for x, y in ls]
    return {
        "init":   np.array(addz(d["uuv_initial_positions"])),
        "target": np.array([d["target_position"][0]*scale,
                            d["target_position"][1]*scale, depth]),
        "wps":    [{k: [x*scale, y*scale, depth] for k, (x, y) in s.items()}
                   for s in d["uuv_targets_per_step"]]
    }

def build_cfg(init, target, sigma=0.0):
    auvs = [ {
        "agent_name": f"auv{i}",
        "agent_type": "HoveringAUV",
        "sensors": [{"sensor_type": "GPSSensor", "socket": "COM",
                     "configuration": {"Sigma": sigma, "Depth": 100}}],
        "control_scheme": 1,
        "location": init[i].tolist(),
        "rotation": [0, 0, 90]
    } for i in range(len(init))]

    torpedo = {
        "agent_name": "torpedo", "agent_type": "TorpedoAUV",
        "sensors": [{"sensor_type": "GPSSensor", "socket": "COM",
                     "configuration": {"Sigma": sigma, "Depth": 100}}],
        "control_scheme": 1, "location": target.tolist(), "rotation": [0, 0, 45]
    }

    leader = {"agent_name": "leader_uuv", "agent_type": "SurfaceVessel",
              "sensors": [], "control_scheme": 1,
              "location": LEADER.tolist(), "rotation": [0, 0, 0]}

    return {"name": "MultiUUVWithLeader", "world": "OpenWater",
            "package_name": "Ocean", "main_agent": "auv0",
            "agents": auvs + [torpedo, leader]}

# ------------ 主仿真 ------------
def simulate(cfg, wps, max_steps=5000, tol=0.1):
    auvs = [a["agent_name"] for a in cfg["agents"] if "auv" in a["agent_name"]]
    prev = {a["agent_name"]: np.array(a["location"]) for a in cfg["agents"]}
    state = {a: "NAVIGATING" for a in auvs}
    idx   = {a: 0 for a in auvs}
    done  = {a: False for a in auvs}

    err_vec  = np.random.normal(0, SIGMA_NAV, 2)
    growth_v = np.zeros(2)

    csv_f = open(os.path.join(OUT_DIR, f"{LOSSY_UUV}_path.csv"), "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow(["t", "ideal_x", "ideal_y", "act_x", "act_y"])

    with holoocean.make(scenario_cfg=cfg) as env:
        torpedo_on = False

        for t in range(max_steps):
            world = env.tick()
            torp  = np.array(world["torpedo"]["GPSSensor"][:3])

            if not torpedo_on and all(done.values()):
                torpedo_on = True

            if torpedo_on:
                acc = TORPEDO_KP * (TORPEDO_GOAL - torp)
                n = np.linalg.norm(acc)
                if n > TORPEDO_MAX:
                    acc *= TORPEDO_MAX / n
                env.act("torpedo", acc.tolist() + [0, 0, 0.002])

            for a in auvs:
                pos = np.array(world[a]["GPSSensor"][:3])
                cmd = pos.copy()
                st  = state[a]

                if st == "NAVIGATING":
                    if a == LOSSY_UUV and torpedo_on and random.random() < LOSS_PROB:
                        state[a] = "LOST"
                        err_vec = np.random.normal(0, SIGMA_LOST0, 2)
                        while np.allclose(err_vec, 0):
                            err_vec = np.random.normal(0, SIGMA_LOST0, 2)
                        growth_v = np.sign(err_vec) * GROWTH_RATE
                        zeros = growth_v == 0
                        if zeros.any():
                            growth_v[zeros] = np.random.choice([-1, 1], size=zeros.sum()) * GROWTH_RATE
                    else:
                        while idx[a] < len(wps) and \
                              wps[idx[a]].get(a.replace("auv", "UUV_")) is None:
                            idx[a] += 1
                        if idx[a] < len(wps):
                            tgt = wps[idx[a]].get(a.replace("auv", "UUV_"))
                            d   = np.array(tgt) - pos
                            yaw = math.degrees(math.atan2(d[1], d[0]))
                            cmd = np.array(tgt)
                            env.draw_point(tgt, thickness=10, lifetime=0)
                            env.act(a, cmd.tolist() + [0, 0, yaw])
                            if np.linalg.norm(d) <= tol:
                                idx[a] += 1
                        else:
                            done[a] = True
                            if torpedo_on:
                                d = torp - pos
                                yaw = math.degrees(math.atan2(d[1], d[0]))
                                cmd = torp
                                env.act(a, cmd.tolist() + [0, 0, yaw])

                        if a == LOSSY_UUV:
                            err_vec = np.random.normal(0, SIGMA_NAV, 2)

                elif st == "LOST":
                    new_z = min(pos[2] + ASCEND_RATE, 0)
                    cmd   = np.array([pos[0], pos[1], new_z])
                    env.act(a, cmd.tolist() + [0, 0, 0])
                    if a == LOSSY_UUV:
                        err_vec += growth_v
                    if new_z >= 0:
                        state[a] = "RECOVERING"

                elif st == "RECOVERING":
                    rng = np.linalg.norm((LEADER - pos)[:2])
                    brg = math.atan2(LEADER[1] - pos[1], LEADER[0] - pos[0])
                    try:
                        est = bearing_range_estimate(LEADER[:2], rng, brg)
                        surf_cmd = np.array([est[0], est[1], 0.0])
                        env.act(a, surf_cmd.tolist() + [0, 0, 0])
                        state[a] = ("DIVING", est[2])
                    except Exception as e:
                        print(f"GTSAM inference failed for {a}: {e}")

                elif isinstance(st, tuple) and st[0] == "DIVING":
                    target_z = st[1]
                    new_z = max(pos[2] - DIVE_RATE, target_z)
                    cmd = np.array([pos[0], pos[1], new_z])
                    env.act(a, cmd.tolist() + [0, 0, 0])
                    if abs(new_z - target_z) < 0.1:
                        state[a] = "NAVIGATING"
                        if a == LOSSY_UUV:
                            err_vec = np.random.normal(0, SIGMA_NAV, 2)

                if a == LOSSY_UUV:
                    if state[a] == "RECOVERING" or (
                        isinstance(state[a], tuple) and state[a][0] == "DIVING"):
                        err_vec *= DECAY

                    ideal_xy = pos[:2] + err_vec
                    writer.writerow([t, ideal_xy[0], ideal_xy[1], pos[0], pos[1]])

                env.draw_line(prev[a].tolist(), pos.tolist(),
                              color=[255 * (a == "auv0"),
                                     255 * (a == "auv1"),
                                     255 * (a == "auv2")],
                              thickness=7, lifetime=0)
                prev[a] = pos

    csv_f.close()

# ------------ 入口 ------------
if __name__ == "__main__":
    data = load_and_preprocess("simulation_results_2d.json")
    cfg  = build_cfg(data["init"], data["target"])
    simulate(cfg, data["wps"])
