# MARLIN: Multi-UUV Acoustic Relative Localization with Variational Inference over Neuralized Factor Graphs

### Authors
Chengfeng Jia, Rong Su, Yun Lu, Jian Gong, Jinde Cao  

### Affiliations
- **[CARTIN, Nanyang Technological University (NTU)](https://www.ntu.edu.sg/cartin):**  
  Centre for Advanced Robotics Technology Innovation, NTU, Singapore.  

- **[Southeast University, School of Mathematics](https://math.seu.edu.cn/jdc/list.htm):**  
  Renowned for mathematical research and interdisciplinary applications.  

- **[Nanjing University of Posts and Telecommunications, College of Automation](https://coa.njupt.edu.cn/):**  
  Focusing on automation, control, and intelligent systems research.  


**Implemented functionality:**  
- Factor graph optimization of multi-UUV trajectories under range/bearing acoustic measurements  
- Physics-informed neural network (Transformer backbone) for modeling time-varying acoustic noise  
- Variational EM algorithm alternating between state graph optimization (M-step) and noise inference (E-step)  
- Demonstration of MARLIN’s localization accuracy under different UUV formations
- **Key Python libraries used:**  
- `numpy`, `scipy`, `matplotlib` – numerical computing and visualization  
- `torch` – training the physics-informed neural network (PINN)  
- `networkx` / `gtsam` (optional) – factor graph representation and optimization  
- `holoocean` – (for simulation) multi-UUV environment setup  
