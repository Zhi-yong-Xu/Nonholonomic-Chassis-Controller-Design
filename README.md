# Nonholonomic Chassis Controller Design Based on Null-Space Behavioral Control
On the basis of the switching control strategy designed for the V1 warehouse, null-space behavioral control is integrated to simultaneously realize trajectory tracking and obstacle avoidance.

## 文件说明
`/model` --- 非完整约束底盘(husky)的mjcf模型

`ZoneCalibration.py` --- 当前场景静摩擦死区标定文件 (用于动力学环的死区补偿)

`requirements.txt` --- 安装依赖

`main.py` --- 主文件，直接 `python main.py` 运行

`/config/husky.yaml` --- 参数配置文件，支持轨迹切换，控制参数配置

`/src` --- 观测器，控制器等代码实现





# 理论说明

# 车辆运动学模型与误差系统

## 1. 运动学模型

### 差速轮式机器人的运动学方程

$$
\dot{q} = \begin{bmatrix}\cos\theta & 0 \\\\ \sin\theta & 0 \\\\ 0 & 1\end{bmatrix}\begin{bmatrix}v \\\\ \omega\end{bmatrix}
$$

其中：

- $q=[x,y,\theta]^T$ 是位姿；
- $v$ 是线速度；
- $\omega$ 是角速度。

## 2. 误差系统建模

设参考位姿 $q_r=[x_r,y_r,\theta_r]^T$，定义全局位置误差：

$$
e_x = x_r - x, \qquad e_y = y_r - y
$$

对全局误差求导：

$$
\begin{cases} \dot{e}_x = \dot{x}_r - \dot{x} = d_x - v\cos\theta \\\\ \dot{e}_y = \dot{y}_r - \dot{y} = d_y - v\sin\theta \end{cases}
$$

其中参考轨迹导数满足：

$$
\dot{x}_r = d_x(t), \quad \dot{y}_r = d_y(t), \quad \dot{\theta}_r = d_\theta(t)
$$

且有界：

$$
\lvert d_x \rvert, \lvert d_y \rvert \le v_{r,max}, \qquad \lvert d_\theta \rvert \le \omega_{r,max}
$$

## 3. 极坐标转换

定义距离与方位角：

$$
\rho = \sqrt{e_x^2 + e_y^2}, \qquad \theta_d = \mathrm{atan2}(e_y, e_x)
$$

满足：

$$
e_x = \rho\cos\theta_d, \qquad e_y = \rho\sin\theta_d
$$

定义**朝向误差**：

$$
\boxed{\alpha = \theta_d - \theta}
$$

## 4. 距离误差运动学

对 $\rho^2 = e_x^2 + e_y^2$ 求导：

$$
2\rho\dot{\rho} = 2e_x\dot{e}_x + 2e_y\dot{e}_y
\;\Rightarrow\;
\dot{\rho} = \frac{e_x\dot{e}_x + e_y\dot{e}_y}{\rho}
$$

代入 $\dot{e}_x,\dot{e}_y$：

$$
\dot{\rho} = \frac{e_x(d_x - v\cos\theta) + e_y(d_y - v\sin\theta)}{\rho}
$$

利用 $e_x/\rho = \cos\theta_d$ 与 $e_y/\rho = \sin\theta_d$：

$$
\dot{\rho} = d_x\cos\theta_d + d_y\sin\theta_d - v(\cos\theta_d\cos\theta + \sin\theta_d\sin\theta)
$$

利用 $\cos\theta_d\cos\theta + \sin\theta_d\sin\theta = \cos(\theta_d - \theta) = \cos\alpha$：

$$
\boxed{\dot{\rho} = d_\rho - v\cos\alpha}
$$

其中等效扰动：

$$
d_\rho = d_x\cos\theta_d + d_y\sin\theta_d, \qquad \lvert d_\rho \rvert \le v_{r,max}
$$

## 5. 方位角运动学

根据 $\frac{d}{dx}\arctan u=\frac{u'}{1+u^2}$，对 $\theta_d = \mathrm{atan2}(e_y, e_x)$ 求导：

$$
\dot{\theta}_d = \frac{e_x\dot{e}_y - e_y\dot{e}_x}{\rho^2}
$$

展开分子：

$$
\begin{aligned} e_x\dot{e}_y - e_y\dot{e}_x &= e_x(d_y - v\sin\theta) - e_y(d_x - v\cos\theta) \\\\ &= e_x d_y - e_y d_x + v(e_y\cos\theta - e_x\sin\theta) \end{aligned}
$$

利用 $e_x = \rho\cos\theta_d$ 与 $e_y = \rho\sin\theta_d$：

$$
e_y\cos\theta - e_x\sin\theta
= \rho(\sin\theta_d\cos\theta - \cos\theta_d\sin\theta)
= \rho\sin(\theta_d - \theta)
= \rho\sin\alpha
$$

因此：

$$
\dot{\theta}_d = \frac{e_x d_y - e_y d_x}{\rho^2} + \frac{v\sin\alpha}{\rho}
$$

定义等效扰动：

$$
d_\theta = \frac{e_x d_y - e_y d_x}{\rho^2}, \qquad \lvert d_\theta \rvert \le \frac{v_{r,max}}{\rho_{min}}
$$

得到：

$$
\boxed{\dot{\theta}_d = d_\theta + \frac{v\sin\alpha}{\rho}}
$$

## 6. 朝向误差运动学

由于 $\alpha = \theta_d - \theta$，求导：

$$
\dot{\alpha} = \dot{\theta}_d - \dot{\theta}
= d_\theta + \frac{v\sin\alpha}{\rho} - \omega
$$

## 7. 误差系统汇总

$$
\begin{cases} \dot{\rho} = d_\rho - v\cos\alpha \\\\ \dot{\alpha} = d_\theta + \dfrac{v\sin\alpha}{\rho} - \omega \end{cases}
$$

# 控制器设计

## 1. 双幂次函数（代码 `sig(x,p)`）

所有控制器都基于双幂次（bi-power）非线性项：

$$
\boxed{\ \mathrm{sig}(x,p) = \lvert x \rvert^{p}\,\mathrm{sign}(x)\ }
$$

性质：

- $0<p<1$：小误差时斜率趋于无穷，实现**有限时间收敛**、消除稳态误差；
- $p>1$：大误差时增长快，加快**远离平衡点**的收敛；
- $p=1$：退化为线性项。

组合两项即得到"近处快速收敛 + 远处快速趋近"的效果。

---

## 2. FTDSO：有限时间干扰观测器（估计 $\dot{q}_r$）

**问题**：前馈航向角需要参考速度 $\dot{q}_r$，但希望只用**位置信息** $q_r$ 而不显式求导。前馈航向角定义为：

$$
\theta_{ff} = \mathrm{atan2}(\dot{y}_r,\ \dot{x}_r)
$$



**方案**：用观测器估计参考位置 $\hat{x}_r,\hat{y}_r$，其导数即为速度估计。

定义观测误差：

$$
e_{ox} = x_r - \hat{x}_r,\qquad e_{oy} = y_r - \hat{y}_r
$$

观测器动态（代码 `K_o1=30, K_o2=20, p_o1=1, p_o2=1.2`）：

$$
\dot{\hat{x}}_r = K_{o1}\,\mathrm{sig}(e_{ox},1) + K_{o2}\,\mathrm{sig}(e_{ox},1.2)
$$

$$
\dot{\hat{y}}_r = K_{o1}\,\mathrm{sig}(e_{oy},1) + K_{o2}\,\mathrm{sig}(e_{oy},1.2)
$$

**稳定性逻辑**：误差动态为

$$
\dot{e}_{ox} = \dot{x}_r - K_{o1}\,\mathrm{sig}(e_{ox},1) - K_{o2}\,\mathrm{sig}(e_{ox},1.2)
$$

$$
\dot{e}_{oy} = \dot{y}_r - K_{o1}\,\mathrm{sig}(e_{oy},1) - K_{o2}\,\mathrm{sig}(e_{oy},1.2)
$$

当增益足够大时 $e_{ox}\to 0$，同理 $e_{oy}\to 0$，从而速度估计收敛到参考速度：

$$
\hat{v}_{rx} = K_{o1}\,\mathrm{sig}(e_{ox},1) + K_{o2}\,\mathrm{sig}(e_{ox},1.2)
$$

$$
\hat{v}_{ry} = K_{o1}\,\mathrm{sig}(e_{oy},1) + K_{o2}\,\mathrm{sig}(e_{oy},1.2)
$$


---

## 3. 外环控制器（级联 FTC 核心）

### 3.1 反馈航向角（指向目标）

$$
\theta_{fb} = \mathrm{atan2}(e_y,\ e_x)
$$

### 3.2 前馈航向角（沿参考运动方向）

$$
\theta_{ff} = \mathrm{atan2}(\hat{v}_{ry},\ \hat{v}_{rx})
$$

### 3.3 航向误差

$$
\alpha = \mathrm{wrap}(\theta_{fb} - \psi),\qquad
\alpha_{ff} = \mathrm{wrap}(\theta_{ff} - \psi)
$$

其中 $\psi$ 为实际偏航角，`wrap_to_pi(x) = atan2(sin x, cos x)` 将角度归一化到 $(-\pi,\pi]$。

### 3.4 模式切换（前向/倒车判据）

当参考点从机器人**后方**经过时，直接转向 $\theta_{fb}$ 需要掉头 180°，产生剧烈抖动。设计切换变量：

$$
\beta = \mathrm{wrap}(\theta_{fb} - \theta_{ff}),\qquad
b = \begin{cases} 0, & \lvert\beta\rvert \le \dfrac{\pi}{2} \quad\text{(前向 FWD)} \\\\ 1, & \lvert\beta\rvert > \dfrac{\pi}{2} \quad\text{(倒车 REV)} \end{cases}
$$

倒车模式下对航向误差加 $\pi$ 修正：

$$
\tilde{\alpha} = \mathrm{wrap}(\alpha - b\pi)
$$

**物理含义**：`b=1` 时若 $\tilde{\alpha}\to 0$，机器人车头指向 $\theta_{fb}+\pi$，即**车尾朝向目标**，通过 $v_c<0$（负线速度）实现倒车跟踪。

### 3.5 位置环（速度幅值指令）

$$
v_d = K_{R1}\,\mathrm{sig}(\rho,\,P_1) + K_{R2}\,\mathrm{sig}(\rho,\,P_2),\qquad P_1=0.5,\ P_2=2.0
$$

沿航向投影为标称线速度：

$$
\boxed{\ v_c = v_d\cos\alpha\ }
$$

- $|\alpha|>\pi/2$（目标在侧后方）→ $\cos\alpha<0$ → 先倒车/减速调整朝向；
- $\alpha\leq >\pi/2$（目标在前方）→ 前进。

### 3.6 姿态环（角速度指令）

$$
\boxed{\ \omega_c = K_{A0}\,\frac{\alpha_{ff}}{\rho} + K_{A1}\,\mathrm{sig}(\rho\tilde{\alpha},\,P_1) + K_{A2}\,\mathrm{sig}(\rho\tilde{\alpha},\,P_2)\ }
$$

## 4. 内环：轮速 PID
外环输出 $(v_c,\omega_c)$ 经 $\Omega$ 得到轮速参考 $\Phi_{ref}=[\omega_R^{ref},\omega_L^{ref}]^\top$。实际轮速 $\Phi=[\phi_R,\phi_L]^\top$ 通过PID控制 + 静摩擦死区补偿跟踪参考速度。

# 展示

<img width="685" height="415" alt="image" src="https://github.com/user-attachments/assets/6c827a57-b1f4-4b84-a408-fa3101b26108" />

