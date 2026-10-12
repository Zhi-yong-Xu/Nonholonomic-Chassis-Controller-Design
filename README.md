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

## 零空间行为控制基本理论

### 基于零空间行为控制的任务设计流程

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

### 多级零空间行为控制链式展开

$$
\dot{q}_{\mathrm{des}}= \dot{q}_1+ N_1 \Big(\dot{q}_2+ N_2 \big(\dot{q}_3 + \cdots + N_{p-2} \big( \dot{q}_{p-1} + N_{p-1} \dot{q}_p \big) \big) \Big)
$$

其中：

$$
\dot{q}_i = J_i^{+} \dot{z}_i^{\mathrm{des}}
$$

$$
N_i = I - J_i^{+} J_i
$$

等价求和形式：

$$
\dot{q}_{\mathrm{des}}
= \sum_{i=1}^{p}
\left( \prod_{k=1}^{i-1} N_k \right) \dot{q}_i
$$

约定：

$$
\prod_{k=1}^{0} N_k = I
$$

因此第一项即为 $\dot{q}_1$。

## 基于零空间行为控制的非完整移动底盘轨迹跟踪 + 避障控制律

### 构型空间，任务空间和控制空间的定义和映射关系

- **构型空间（状态空间）**：$`\boldsymbol{q} \in \mathbb{R}^3`$ ，任务函数的自变量

$$
\boldsymbol{q} = \begin{bmatrix} x \\\\ y \\\\ \theta \end{bmatrix} \in \mathbb{R}^3
$$


- **任务空间**：$`\mathbf{z}_i = h_i(\mathbf{q}) \in \mathbb{R}^{m_i}`$，第 $i$ 个任务函数的像

- **控制空间**：$`\boldsymbol{v} \in \mathbb{R}^2`$，底盘车体坐标速度

$$
\boldsymbol{v} = \begin{bmatrix} u \\\\ \omega \end{bmatrix} \in \mathbb{R}^2
$$

构型空间速度与控制空间之间由**输出映射** $R(\theta)$ 联系：

$$
\dot{\boldsymbol{q}} = R(\theta)\,\boldsymbol{v}, \qquad
R(\theta) = \begin{bmatrix} \cos\theta & 0 \\\\ \sin\theta & 0 \\\\ 0 & 1 \end{bmatrix} \in \mathbb{R}^{3\times 2}
$$

由于 $R(\theta)$ 列满秩且列向量正交归一，其最小二乘逆即转置：

$$
R^{+}(\theta) = R^T(\theta) = \begin{bmatrix} \cos\theta & \sin\theta & 0 \\\\ 0 & 0 & 1 \end{bmatrix}
$$

因此给定构型空间的期望速度 $\dot{\boldsymbol{q}}_{cmd}$，可实现的速度指令为它在 $R(\theta)$ 列空间上的投影：

$$
\boldsymbol{v} = R^{+}(\theta)\,\dot{\boldsymbol{q}}_{cmd}, \qquad
\dot{\boldsymbol{q}} = R(\theta)R^{+}(\theta)\,\dot{\boldsymbol{q}}_{cmd} \equiv P_R(\theta)\,\dot{\boldsymbol{q}}_{cmd}
$$

投影算子

$$
P_R(\theta) = R R^T = \begin{bmatrix} \cos^2\theta & \sin\theta\cos\theta & 0 \\\\ \sin\theta\cos\theta & \sin^2\theta & 0 \\\\ 0 & 0 & 1 \end{bmatrix}
$$

把期望速度分解为“沿航向的平移分量 + 航向转动分量”，**垂直于航向的侧向分量被投影消除**——这正是非完整约束 $\dot{x}\sin\theta - \dot{y}\cos\theta = 0$ 在控制层面的体现。

**Tip:** 非完整约束底盘是欠驱动系统，故任务空间的合成速度不一定可达，这是和机械臂这类全驱动系统最本质的区别。

### 轨迹跟踪任务函数和雅可比矩阵

轨迹跟踪任务直接以完整构型为任务变量（**恒等任务**）：

$$
h_m(\boldsymbol{q}) = \boldsymbol{q}, \qquad
\boldsymbol{z}_m = \begin{bmatrix} x \\\\ y \\\\ \theta \end{bmatrix}, \qquad
J_m = \frac{\partial h_m}{\partial \boldsymbol{q}} = I_3 \in \mathbb{R}^{3\times 3}
$$

注意：$`J_m = I_3`$ 意味着该任务的零空间为 $\{\boldsymbol{0}\}$.

由于本项目的控制律在V1中已经设计完成（计为$`\boldsymbol{v}_m = [ u_m; \omega_m]`$），故对应的任务空间速度通过

$$
\dot{\boldsymbol{q}}_m = R(\theta)\,\boldsymbol{v}_m
$$

获得。

### 避障任务函数和雅可比矩阵

#### 任务函数

设障碍物边界点集为 $\partial\mathcal{O}$，先取最近点：

$$
\boldsymbol{p}_o = \arg\min_{\boldsymbol{p}_j \in \partial\mathcal{O}} \|\boldsymbol{p} - \boldsymbol{p}_j\|^2, \qquad
d(\boldsymbol{q}) = \|\boldsymbol{p} - \boldsymbol{p}_o\|, \qquad
\hat{\boldsymbol{u}} = \frac{\boldsymbol{p} - \boldsymbol{p}_o}{d}
$$


避障任务函数取二维向量：

$$
h_o(\boldsymbol{q}) = \begin{bmatrix} d(\boldsymbol{q}) \\\\ \theta(\boldsymbol{q}) \end{bmatrix} \in \mathbb{R}^2,
\qquad \boldsymbol{z}_o = \begin{bmatrix} d \\\\ \theta \end{bmatrix}
$$

即“到障碍物的径向距离 + 当前航向”。

#### 雅可比矩阵

对 $d$ 求梯度：$`\nabla_{\boldsymbol{p}} d = \hat{\boldsymbol{u}}^T`$；对 $\theta$ 求梯度为 $[0\; 0\; 1]$。故：

$$
J_o(\boldsymbol{q}) = \begin{bmatrix} \dfrac{\partial d}{\partial x} & \dfrac{\partial d}{\partial y} & \dfrac{\partial d}{\partial \theta} \\\\[6pt] \dfrac{\partial \theta}{\partial x} & \dfrac{\partial \theta}{\partial y} & \dfrac{\partial \theta}{\partial \theta} \end{bmatrix}
= \begin{bmatrix} \hat{u}_x & \hat{u}_y & 0 \\\\ 0 & 0 & 1 \end{bmatrix} \in \mathbb{R}^{2\times 3}
$$
其两行单位正交，因此伪逆有解析形式 $J_o^{+} = J_o^T$：
$$
J_o^{+} = \begin{bmatrix} \hat{u}_x & 0 \\ \hat{u}_y & 0 \\ 0 & 1 \end{bmatrix}
$$
### 3.3 期望任务变量与逃逸航向
期望任务变量为“安全距离 + 切向逃逸航向”：
$$
\boldsymbol{z}_o^{d} = \begin{bmatrix} d_{safe} \\ \theta_d \end{bmatrix}
$$
其中 $\theta_d$ 把“指向障碍物”的方向旋转 $\pm 90°$，绕行侧依据智能体相对障碍物的纵向位置：
$$
\theta_d = \operatorname{atan2}(y_o - y,\; x_o - x) + \sigma\frac{\pi}{2}, \qquad
\sigma = \begin{cases} +1, & y > y_o \\ -1, & y \leq y_o \end{cases}
$$
### 3.4 任务速度指令
任务空间误差做比例控制，再经 $J_o^{+}$ 映射到构型空间：
$$
\dot{\boldsymbol{q}}_o = J_o^{+}\, k_o\left(\boldsymbol{z}_o^{d} - \boldsymbol{z}_o\right), \qquad k_o = 20
$$
展开为分量形式：
$$
\boxed{\;\dot{\boldsymbol{q}}_o = k_o \begin{bmatrix} \hat{u}_x\,\left(d_{safe} - d\right) \\[3pt] \hat{u}_y\,\left(d_{safe} - d\right) \\[3pt] \theta_d - \theta \end{bmatrix}\;}
$$
物理意义：
- **径向排斥分量**：沿 $\hat{\boldsymbol{u}}$（离开障碍物方向）施加速度，幅值正比于距离亏缺 $d_{safe} - d$，将距离镇定到 $d_{safe}$；
- **航向分量**：比例镇定航向到切向逃逸角 $\theta_d$，使机器人沿障碍物边界滑行而非正面冲撞。

### 任务切换规则设计与零空间合成



# 展示

<img width="685" height="415" alt="image" src="https://github.com/user-attachments/assets/6c827a57-b1f4-4b84-a408-fa3101b26108" />

