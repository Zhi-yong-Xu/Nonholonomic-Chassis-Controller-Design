# Nonholonomic Chassis Controller Design Based on Null-Space Behavioral Control
On the basis of the switching control strategy designed for the V1 warehouse, null-space behavioral control is integrated to simultaneously realize trajectory tracking and obstacle avoidance.

## 文件说明
`/model` --- 非完整约束底盘(husky)的mjcf模型

`requirements.txt` --- 安装依赖

`main.py` --- 主文件，直接 `python main.py` 运行

`/config/husky.yaml` --- 参数配置文件，支持轨迹切换，控制参数配置

`/src` --- 观测器，控制器等代码实现





# 理论说明



# 轨迹跟踪切换控制器设计见V1版本的 `Readme.md`

# NSBC基本理论

1. **任务函数设计**：设计 $i$ 个任务定义任务函数 $z_i = f_i(q) \in \mathbb{R}^{m_i}$，它是关于 $q$ 的可测函数，并给定期望值 $z_i^*$。
2. **求解任务函数的雅可比**：定义任务雅可比矩阵

    $$
    J_i \triangleq \frac{\partial f_i}{\partial q} \in \mathbb{R}^{m_i \times 3}
    $$
    
    得到 $\dot z_i = J_i\,\dot q$。

3. **设计任务控制率**：
   
   $$
    \dot z_i^{des} = \dot z_i^* - \Lambda_i(z_i)\,\big(z_i - z_i^*\big)
   $$

   其中 $\Lambda_i$ 为增益

4. **任务速度映射**：解 $J_i\dot q_i = \dot z_i^{\mathrm{des}}$ 的**最小范数解**：
    $$
    \boxed{\;\dot q_i = J_i^{+}\,\dot z_i^{\mathrm{des}},\qquad
    J_i^{+} = J_i^T\big(J_iJ_i^T\big)^{-1}\;}\quad(J_i\text{ 满行秩时})
    $$
    它满足 $J_i\dot q_i = \dot z_i^{\mathrm{des}}$（任务速率精确实现），且在所有可行实现中位姿速度范数最小。

# 展示

<img width="685" height="415" alt="image" src="https://github.com/user-attachments/assets/6c827a57-b1f4-4b84-a408-fa3101b26108" />

