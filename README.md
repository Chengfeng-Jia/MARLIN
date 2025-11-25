# MARLIN: Multi-UUV Acoustic Relative Localization via Variational Inference on Neuralized Factor Graphs

**Implemented functionality:**  
- Factor graph optimization of multi-UUV trajectories under range/bearing acoustic measurements  
- Physics-informed neural network (Transformer backbone) for modeling time-varying acoustic noise  
- Variational EM algorithm alternating between state graph optimization (M-step) and noise inference (E-step)  
- Demonstration of MARLIN’s localization accuracy under different UUV formations
  
 **Key Python libraries used:**  
- `numpy`, `scipy`, `matplotlib` – numerical computing and visualization  
- `torch` – training the physics-informed neural network (PINN)  
- `networkx` / `gtsam` (optional) – factor graph representation and optimization  
- `holoocean` – (for simulation) multi-UUV environment setup  
