import math, numpy as np, pandas as pd, gtsam, torch, torch.nn as nn
from   gtsam import symbol, Pose2, Rot2, noiseModel, Marginals
from   gtsam.noiseModel import Robust, mEstimator

# ───────── Hyper-parameters ─────────
EM_ITERS, INNER_STEPS, MC_SAMPLES = 10, 10, 1024
LR, GRAD_CLIP                     = 5e-4, 5.0
ALPHA, SIGMA_Z, BETA_BASE         = 3.0, 0.12, 2.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────── Noise models ─────────
prior_leader = noiseModel.Diagonal.Sigmas([1e-4]*3)
motion_noise = noiseModel.Diagonal.Sigmas([0.05,0.05,0.01])

range_noise  = noiseModel.Isotropic.Sigma(1, SIGMA_Z)
robust_range = Robust.Create(mEstimator.Huber(1.345), range_noise)

bearing_sigma = 0.06
bearing_noise = noiseModel.Isotropic.Sigma(1, bearing_sigma)

lm_params = gtsam.LevenbergMarquardtParams()
lm_params.setVerbosityLM("ERROR")

# ───────── Factor-graph builder ─────────
def build_graph(df: pd.DataFrame, bias=None):
    g, init = gtsam.NonlinearFactorGraph(), gtsam.Values()
    for k, row in df.iterrows():
        L, F = symbol("L", k), symbol("F", k)

        if not init.exists(L):
            lp = Pose2(row.leader1_x, row.leader1_y, row.leader1_theta)
            init.insert(L, lp)
            g.add(gtsam.PriorFactorPose2(L, lp, prior_leader))

        if k == 0:
            fp = Pose2(row.follower_x, row.follower_y, row.follower_theta)
            init.insert(F, fp)
            g.add(gtsam.PriorFactorPose2(
                F, fp, noiseModel.Diagonal.Sigmas([0.01,0.01,0.02])))
        else:
            odom = Pose2(row.delta_x_odom,row.delta_y_odom,row.delta_theta_odom)
            g.add(gtsam.BetweenFactorPose2(symbol("F",k-1), F, odom, motion_noise))
            prev = init.atPose2(symbol("F",k-1))
            init.insert(F, prev.compose(odom))

        rng = row.distance_noisy_1 - (0.0 if bias is None else bias[k])
        g.add(gtsam.RangeFactorPose2(L, F, rng, robust_range))
        g.add(gtsam.BearingFactorPose2(L, F, Rot2(row.bearing_noisy_1), bearing_noise))
    return g, init

# ───────── Transformer β model (no priors) ─────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):            # x: (1,T,d_model)
        return x + self.pe[:,:x.size(1)]

class BetaTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=4, nlayers=3):
        super().__init__()
        self.in_proj  = nn.Linear(2, d_model)              # features: [τ,d_pred]
        self.pos_enc  = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model*4, batch_first=True, dropout=0.0)
        self.encoder  = nn.TransformerEncoder(layer, nlayers)
        self.out_proj = nn.Linear(d_model, 1)
    def forward(self, tau, d_pred):                        # tau,d_pred: (T,)
        x = torch.stack([tau, d_pred], dim=-1).unsqueeze(0)  # (1,T,2)
        x = self.in_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x).squeeze(0)                     # (T,d_model)
        beta = torch.nn.functional.softplus(
                   self.out_proj(x).squeeze(-1)) + 1e-6    # ensure β>0
        return beta                                        # (T,)

# ───────── ELBO ─────────
def neg_elbo(resid, beta_pred):
    a = torch.full_like(beta_pred, ALPHA)
    q = torch.distributions.InverseGamma(a, beta_pred)
    R = q.rsample((MC_SAMPLES,))                           # (M,T)

    log_pz = -0.5*((resid.unsqueeze(0)-R)/SIGMA_Z)**2 \
             - math.log(math.sqrt(2*math.pi)*SIGMA_Z)

    beta_prior = torch.full_like(beta_pred, BETA_BASE)     # flat-ish weak prior
    p = torch.distributions.InverseGamma(a, beta_prior)

    elbo = (log_pz + p.log_prob(R) - q.log_prob(R)).mean()
    return -elbo

# ───────── Main ─────────
if __name__ == "__main__":
    df = pd.read_csv("uuv_motion_data_gamma1.csv")
    T  = len(df)

    model = BetaTransformer().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    tau_t = torch.tensor(np.arange(T)/(T-1), dtype=torch.float32, device=device)

    graph, init = build_graph(df)
    result = gtsam.LevenbergMarquardtOptimizer(graph, init, lm_params).optimize()

    for em in range(EM_ITERS):
        # ---------- predicted ranges ----------
        d_pred = np.array([
            result.atPose2(symbol("F",k)).range(
                [row.leader1_x,row.leader1_y])
            for k,row in df.iterrows()
        ], dtype=np.float32)

        resid_t  = torch.tensor(df.distance_noisy_1.values - d_pred,
                                dtype=torch.float32, device=device)
        d_pred_t = torch.tensor(d_pred, device=device)

        # ---------- update β ----------
        for _ in range(INNER_STEPS):
            beta_pred_t = model(tau_t, d_pred_t).clamp(min=1e-3)
            loss = neg_elbo(resid_t, beta_pred_t)
            optim.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()

        with torch.no_grad():
            beta_pred_t = model(tau_t, d_pred_t)
            Rhat = (beta_pred_t / (ALPHA-1)).cpu().numpy()

        # 3-pt moving-average to calm jitter
        Rhat_sm = pd.Series(Rhat).rolling(3, center=True,
                                          min_periods=1).mean()

        print(f"EM{em:02d}  -ELBO={loss.item():.3f}  "
              f"mean β={beta_pred_t.mean():.3f}")

        graph, init = build_graph(df, Rhat_sm)
        result = gtsam.LevenbergMarquardtOptimizer(graph, init, lm_params).optimize()

    # ---------- save model ----------
    torch.save(model.state_dict(), "noprior.pth")

    # ---------- export follower pose & β ----------
    marg, rows = Marginals(graph, result), []
    beta_mean  = (beta_pred_t/(ALPHA-1)).cpu().numpy()
    for k in range(T):
        p   = result.atPose2(symbol("F",k))
        cov = marg.marginalCovariance(symbol("F",k))
        rows.append(dict(
            x=p.x(), y=p.y(), theta=p.theta(),
            std_x=math.sqrt(cov[0,0]), std_y=math.sqrt(cov[1,1]),
            std_theta=math.sqrt(cov[2,2]),
            R_mean=float(beta_mean[k]), alpha=ALPHA,
            beta=float(beta_pred_t[k].cpu())
        ))
    pd.DataFrame(rows).to_csv("NoPrior.csv", index=False)
    print("Done")
