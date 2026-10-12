# Nonholonomic Chassis Controller Design Based on Null-Space Behavioral Control
On the basis of the switching control strategy designed for the V1 warehouse, null-space behavioral control is integrated to simultaneously realize trajectory tracking and obstacle avoidance.

## 文件说明
`/model` --- 非完整约束底盘(husky)的mjcf模型

`requirements.txt` --- 安装依赖

`main.py` --- 主文件，直接 `python main.py` 运行

`/config/husky.yaml` --- 参数配置文件，支持轨迹切换，控制参数配置

`/src` --- 观测器，控制器等代码实现





# 理论说明



## 轨迹跟踪切换控制器设计见V1版本的 `Readme.md`

## 基于零空间行为控制的任务设计流程

1. **任务函数设计**：设计 $i$ 个任务，定义任务函数

$$
   z_i = f_i(q) \in \mathbb{R}^{m_i}
$$

   其中 $z_i$ 是关于 $q$ 的可测函数，并给定期望值 $z_i^*$。

2. **求解任务函数的雅可比矩阵**：定义任务雅可比矩阵
   $$
   J_i \triangleq \frac{\partial f_i}{\partial q} \in \mathbb{R}^{m_i \times 3}
   $$
   从而得到
   $$
   \dot{z}_i = J_i \dot{q}
   $$

3. **设计任务控制律**：定义期望任务速度
   $$
   \dot{z}_i^{\mathrm{des}} = \dot{z}_i^* - \Lambda_i(z_i)\big(z_i - z_i^*\big)
   $$
   其中 $\Lambda_i$ 为增益矩阵。

4. **任务速度映射**：求解
   $$
   J_i \dot{q}_i = \dot{z}_i^{\mathrm{des}}
   $$
   的最小范数解：
   $$
   \dot{q}_i = J_i^{+} \dot{z}_i^{\mathrm{des}}
   $$
   其中
   $$
   J_i^{+} = J_i^T\big(J_i J_i^T\big)^{-1}
   $$
   该伪逆在 $J_i$ 满行秩时成立。此时满足
   $$
   J_i \dot{q}_i = \dot{z}_i^{\mathrm{des}}
   $$
   即任务速率被精确实现，且在所有可行解中 $\dot{q}_i$ 的范数最小。

5. **优先级融合与逆映射**：按照任务优先级对各任务速度进行融合，得到总的期望广义速度 $\dot{q}_{\mathrm{des}}$，再通过底盘运动学逆映射得到底盘控制输入 $(v,\omega)$。

# 展示

<img width="685" height="415" alt="image" src="https://github.com/user-attachments/assets/6c827a57-b1f4-4b84-a408-fa3101b26108" />

