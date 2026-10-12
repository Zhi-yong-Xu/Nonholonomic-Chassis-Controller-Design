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

1. **任务函数设计**：设计 $i$ 个任务定义任务函数 $ z_i = f_i(q) \in \mathbb{R}^{m_i}$ , 并给定期望值 $z_i^*$。


# 展示

<img width="685" height="415" alt="image" src="https://github.com/user-attachments/assets/6c827a57-b1f4-4b84-a408-fa3101b26108" />

