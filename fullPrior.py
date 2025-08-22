import math
import numpy as np
import pandas as pd
import gtsam
import torch
import torch.nn as nn
import torch.nn.functional as F
from gtsam import symbol, Pose2, Rot2, noiseModel, Marginals
from gtsam.noiseModel import Robust, mEstimator

# ───── Hyperparameters ─────
EM_ITERS        = 10
INNER_STEPS     = 10
MC_SAMPLES      = 512
LR              = 1e-3
GRAD_CLIP       = 5.0
ALPHA           = 3.0
SIGMA_Z         = 0.12
BETA_BASE       = 2.0
D0              = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───── GTSAM noise models ─────
prior_leader  = noiseModel.Diagonal.Sigmas([1e-4] * 3)
motion_noise  = noiseModel.Diagonal.Sigmas([0.05, 0.05, 0.01])
range_noise   = noiseModel.Isotropic.Sigma(1, SIGMA_Z)
bearing_noise = noiseModel.Isotropic.Sigma(1, 0.06)

# ───── Robust Huber for range ─────
huber = mEstimator.Huber(1.345)
robust_range = Robust.Create(huber, range_noise)

# ───── Levenberg–Marquardt params ─────
lm_params = gtsam.LevenbergMarquardtParams()
lm_params.setVerbosityLM("ERROR")
lm_params.setMaxIterations(50)
lm_params.setRelativeErrorTol(1e-5)

def build_graph(df, bias=None):
    graph, init = gtsam.NonlinearFactorGraph(), gtsam.Values()
    for idx, row in df.iterrows():
        L, F = symbol("L", idx), symbol("F", idx)
        # Leader prior
        if not init.exists(L):
            lp = Pose2(row.leader1_x, row.leader1_y, row.leader1_theta)
            init.insert(L, lp)
            graph.add(gtsam.PriorFactorPose2(L, lp, prior_leader))
        # Follower prior or odom factor
        if idx == 0:
            fp = Pose2(row.follower_x, row.follower_y, row.follower_theta)
            init.insert(F, fp)
            graph.add(gtsam.PriorFactorPose2(
                F, fp,
                noiseModel.Diagonal.Sigmas([0.05, 0.05, 0.01])
            ))
        else:
            prev = init.atPose2(symbol("F", idx - 1))
            fp = Pose2(
                prev.x() + row.delta_x_odom,
                prev.y() + row.delta_y_odom,
                prev.theta() + row.delta_theta_odom
            )
            graph.add(gtsam.BetweenFactorPose2(
                symbol("F", idx - 1), F,
                Pose2(row.delta_x_odom, row.delta_y_odom, row.delta_theta_odom),
                motion_noise
            ))
            init.insert(F, fp)

        # Range & bearing measurements (using robust range)
        bias_t = 0.0 if bias is None else float(bias[idx])
        rng_meas = float(row.distance_noisy_1) - bias_t
        graph.add(gtsam.RangeFactorPose2(L, F, rng_meas, robust_range))
        graph.add(gtsam.BearingFactorPose2(
            L, F, Rot2(row.bearing_noisy_1), bearing_noise
        ))
    return graph, init

# ───── Positional Encoding ─────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x * math.sqrt(x.size(-1))
        return x + self.pe[: x.size(0)]

# ───── Transformer β Estimator ─────
class BetaTransformer(nn.Module):
    def __init__(self, d_model=8, nhead=1, nlayers=1):
        super().__init__()
        self.in_proj = nn.Linear(2, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=16, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.out_proj = nn.Linear(d_model, 1)
        self.theta_time = nn.Parameter(torch.tensor(0.30))
        self.theta_dist = nn.Parameter(torch.tensor(1.50))
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, tau_t, d_pred_t):
        feat = torch.stack([tau_t, d_pred_t], dim=1)
        x = self.in_proj(feat)
        x = self.pos_enc(x)
        x = x.unsqueeze(0)
        x = self.encoder(x).squeeze(0)
        delta = 1.0 + F.softplus(self.out_proj(x).squeeze(-1))
        delta = torch.clamp(delta, min=1.0, max=2.0)
        beta_time = 1 + torch.clamp(self.theta_time, -5, 10) * tau_t
        dist_scale = 1 + torch.clamp(self.theta_dist, -5, 10) * torch.log(d_pred_t / D0 + 1.0)
        beta_skel = BETA_BASE * beta_time * dist_scale
        beta_pred = beta_skel * delta
        return beta_pred.clamp(min=1e-3), beta_skel

def neg_elbo(resid, beta_pred, beta_prior):
    a = torch.full_like(beta_pred, ALPHA)
    R = torch.distributions.InverseGamma(a, beta_pred).rsample((MC_SAMPLES,))
    log_q = torch.distributions.InverseGamma(a, beta_pred).log_prob(R)
    log_pz = (-0.5 * ((resid.unsqueeze(0) - R) / SIGMA_Z) ** 2
              - math.log(math.sqrt(2.0 * math.pi)) - math.log(SIGMA_Z))
    log_pr = torch.distributions.InverseGamma(a, beta_prior).log_prob(R)
    return -torch.mean(log_pz + log_pr - log_q)

# ───── Main Training ─────
if __name__ == "__main__":
    df = pd.read_csv("uuv_motion_data_gamma1.csv")
    T = len(df)

    model = BetaTransformer().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    # initial graph & optimize
    graph, init = build_graph(df)
    result = gtsam.LevenbergMarquardtOptimizer(graph, init, lm_params).optimize()

    for em in range(EM_ITERS):
        d_pred, tau = [], []
        for k, row in df.iterrows():
            pf = result.atPose2(symbol("F", k))
            d_pred.append(pf.range([row.leader1_x, row.leader1_y]))
            tau.append(k / (T - 1))
        tau = torch.tensor(tau, device=device)
        d_pred = torch.tensor(d_pred, device=device)
        resid = torch.tensor(df.distance_noisy_1.values - d_pred.cpu().numpy(),
                             device=device)

        with torch.no_grad():
            _, beta_prior_t = model(tau, d_pred)

        for p in (model.theta_time, model.theta_dist):
            p.requires_grad = em >= 2

        for _ in range(INNER_STEPS):
            beta_pred_t, _ = model(tau, d_pred)
            beta_pred_t = beta_pred_t.clamp(min=2.0, max=20.0)
            loss = neg_elbo(resid, beta_pred_t, beta_prior_t)
            smoothness = torch.sum((beta_pred_t[2:] - 2*beta_pred_t[1:-1] + beta_pred_t[:-2])**2)
            loss = loss + 0.01 * smoothness

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()

        with torch.no_grad():
            beta_pred_t, _ = model(tau, d_pred)
            Rhat = (beta_pred_t/(ALPHA - 1)).flatten().cpu().numpy()
            Rhat = np.clip(Rhat, 0.5, 10.0)  # 可选：截断异常值
            assert len(Rhat) == len(df), f"Rhat length {len(Rhat)} does not match df {len(df)}"

        print(f"EM{em:02d}  -ELBO={loss.item():.4f}  mean β={beta_pred_t.mean():.3f}")

        graph, init = build_graph(df, pd.Series(Rhat, index=df.index))
        result = gtsam.LevenbergMarquardtOptimizer(graph, init, lm_params).optimize()

    # Save model & export results
    torch.save(model.state_dict(), "fullprior.pth")
    marg = Marginals(graph, result)

    rows = []
    for k in range(T):
        p = result.atPose2(symbol("F", k))
        cov = marg.marginalCovariance(symbol("F", k))
        rows.append({
            "x": p.x(), "y": p.y(), "theta": p.theta(),
            "std_x": math.sqrt(cov[0, 0]), "std_y": math.sqrt(cov[1, 1]),
            "std_theta": math.sqrt(cov[2, 2]),
            "R_mean": float((beta_pred_t[k] / (ALPHA - 1)).cpu()),
            "alpha": ALPHA, "beta": float(beta_pred_t[k].cpu())
        })
    pd.DataFrame(rows).to_csv("FullPrior.csv", index=False)
    print("Done")
