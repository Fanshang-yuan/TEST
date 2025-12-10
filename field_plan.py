import numpy as np
import matplotlib.pyplot as plt

# --- 1. 参数设置 ---
x_range = np.linspace(-2, 12, 100)
y_range = np.linspace(-2, 12, 100)
X, Y = np.meshgrid(x_range, y_range)

# 目标位置
goal = np.array([10, 10])
# 静态障碍物 (位置, 峰值A, 扩散范围sigma)
static_obs = {'pos': np.array([5, 5]), 'A': 50.0, 'sigma': 1.5}
# 动态/不确定障碍物 (位置, 峰值A, 扩散范围sigma - 范围更大)
dynamic_obs = {'pos': np.array([7, 6]), 'A': 40.0, 'sigma': 2.5} 

# --- 2. 场计算函数 ---

def calc_attractive_potential(X, Y, goal, k=1.5):
    # 计算到目标的距离作为引力势能
    return 0.5 * k * np.sqrt((X - goal[0])**2 + (Y - goal[1])**2)

def calc_risk_field(X, Y, obs):
    # 高斯分布公式计算风险场
    d2 = (X - obs['pos'][0])**2 + (Y - obs['pos'][1])**2
    return obs['A'] * np.exp(-d2 / (2 * obs['sigma']**2))

# --- 3. 计算总场 ---
U_att = calc_attractive_potential(X, Y, goal)
R_static = calc_risk_field(X, Y, static_obs)
R_dynamic = calc_risk_field(X, Y, dynamic_obs)

# 总势场 = 引力 + 风险
U_total = U_att + R_static + R_dynamic

# --- 4. 模拟路径规划 (梯度下降法) ---
# 这是一个简化的规划器，真实场景会用 MPC 或 A*
path = []
current_pos = np.array([0.0, 0.0]) # 起点
path.append(current_pos.copy())

lr = 0.1 # 步长
for _ in range(200):
    
    d_x = current_pos[0] - goal[0] # 引力分量
    d_y = current_pos[1] - goal[1]
    
    # 加上障碍物的排斥力 (梯度的反方向)
    for obs in [static_obs, dynamic_obs]:
        dist_sq = np.sum((current_pos - obs['pos'])**2)
        # 高斯函数的导数
        repulsive_force = (current_pos - obs['pos']) * (obs['A'] / obs['sigma']**2) * np.exp(-dist_sq / (2 * obs['sigma']**2))
        d_x += repulsive_force[0]
        d_y += repulsive_force[1]
        
    # 归一化并移动
    grad = np.array([d_x, d_y])
    grad = grad / (np.linalg.norm(grad) + 0.01) # 防止除零
    
    current_pos -= lr * grad # 沿梯度的反方向（下坡）走
    path.append(current_pos.copy())
    
    if np.linalg.norm(current_pos - goal) < 0.2:
        break

path = np.array(path)

# --- 5. 绘图 ---
plt.figure(figsize=(10, 8))
# 绘制等高线 (地形图)
contour = plt.contourf(X, Y, U_total, levels=50, cmap='viridis')
plt.colorbar(contour, label='Risk Potential (Cost)')

# 绘制起点、终点、障碍物
plt.scatter(0, 0, c='green', s=100, label='Start')
plt.scatter(goal[0], goal[1], c='red', marker='*', s=200, label='Goal')
plt.scatter(static_obs['pos'][0], static_obs['pos'][1], c='white', marker='x', label='Static Obs')
plt.scatter(dynamic_obs['pos'][0], dynamic_obs['pos'][1], c='orange', marker='x', label='Dynamic Obs (Uncertain)')

# 绘制规划出的路径
plt.plot(path[:,0], path[:,1], c='white', linewidth=2, linestyle='--', label='Planned Path')

plt.title("Robot Motion Planning based on Risk Field")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()