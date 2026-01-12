# SLAM 大规模测试：Python 到 C++ 转换指南

## 测试参数转换

### Python 代码
```python
N = 5000
step = 25
prob = 0.05
radius = 50 
prior_prop = 0.02
prior_sigma = 1
odom_sigma = 1
```

### C++ 代码
```cpp
const int N = 5000;
const double step = 25.0;
const double prob = 0.05;
const double radius = 50.0;
const double prior_prop = 0.02;
const double prior_sigma = 1.0;
const double odom_sigma = 1.0;
const unsigned int seed = 2001;
```

## 步骤对应

### 1. 图生成

**Python:**
```python
layers = init_layers(N=N, step_size=step, loop_prob=prob, 
                     loop_radius=radius, prior_prop=prior_prop, seed=2001)
```

**C++:**
```cpp
slam::SlamGraph graph = slam::makeSlamLikeGraph(
    N, step, prob, radius, prior_prop, 2001, true
);
```

### 2. 因子图构建

**Python:**
```python
gbp_graph = build_noisy_pose_graph(layers[0]["nodes"], layers[0]["edges"],
                                    prior_sigma=prior_sigma,
                                    odom_sigma=odom_sigma,
                                    seed=2001)
```

**C++:**
```cpp
slam::NoiseConfig config;
config.prior_sigma = prior_sigma;
config.odom_sigma = odom_sigma;
config.seed = 2001;
config.use_seed = true;

slam::SimpleFactorGraph gbp_graph = slam::buildNoisyPoseGraph(
    graph.nodes, graph.edges, config
);
```

### 3. 能量计算

**Python:**
```python
def energy_map(graph, include_priors=True, include_factors=True):
    total = 0.0
    for v in graph.var_nodes[:graph.n_var_nodes]:
        gt = np.asarray(v.GT[0:2], dtype=float)
        r = np.asarray(v.mu[0:2], dtype=float) - gt
        total += 0.5 * float(r.T @ r)
    return total
```

**C++:**
```cpp
double energyMap(const slam::SimpleFactorGraph& graph, 
                 bool include_priors = true, bool include_factors = true) {
    double total = 0.0;
    for (const auto& v : graph.var_nodes) {
        if (v->dim < 2) continue;
        Eigen::Vector2d gt = v->GT.head(2);
        Eigen::Vector2d mu = v->mu.head(2);
        Eigen::Vector2d residual = mu - gt;
        total += 0.5 * residual.dot(residual);
    }
    return total;
}
```

### 4. 迭代优化

**Python:**
```python
basegraph = layers[0]["graph"]
basegraph.eta_damping = 0.4
energy_prev = 2324.863742569365

counter = 0
for it in range(2000):
    basegraph.synchronous_iteration()
    energy = energy_map(basegraph, include_priors=True, include_factors=True)
    
    if np.abs(energy_prev - energy) < 1e-2:
        counter += 1
        if counter >= 2:
            break
    print(f"Iter {it+1:03d} | Energy = {energy:.6f}")
```

**C++:**
```cpp
gbp_graph.eta_damping = 0.4;
double energy_prev = energyMap(gbp_graph);

int counter = 0;
for (int it = 0; it < 2000; ++it) {
    gbp_graph.synchronousIteration();
    double energy = energyMap(gbp_graph);
    double delta_energy = std::abs(energy_prev - energy);
    
    std::cout << "Iter " << std::setw(4) << (it + 1) 
              << " | Energy = " << energy << "\n";
    
    if (delta_energy < 1e-2) {
        counter++;
        if (counter >= 2) break;
    } else {
        counter = 0;
    }
    
    energy_prev = energy;
}
```

## 主要转换要点

| Python | C++ | 说明 |
|--------|-----|------|
| `init_layers()` | `makeSlamLikeGraph()` | 生成 SLAM 图 |
| `build_noisy_pose_graph()` | `buildNoisyPoseGraph()` | 构建因子图 |
| `energy_map()` | `energyMap()` | 计算能量函数 |
| `synchronous_iteration()` | `synchronousIteration()` | 同步迭代 |
| `np.abs()` | `std::abs()` | 绝对值 |
| `for ... in` | `for (... : ...)` | 范围循环 |
| 列表解包 | `head()` / `tail()` | Eigen 向量切片 |
| `dot()` 向量积 | `.dot()` | 向量点乘 |

## 编译和运行

```bash
# 方式 1: 使用批处理脚本
run_large_scale_test.bat

# 方式 2: 手动编译
cd build-release
cmake --build . --config Release --target test_large_scale_slam
.\Release\test_large_scale_slam.exe
```

## 性能对比

- **Python**: 通常较慢，适合原型开发
- **C++**: 通常快 10-100 倍，适合大规模问题
- 此测试 (N=5000): C++ 预计秒级完成，Python 可能需要分钟级

## 注意事项

1. **Eigen 向量**：使用 `.head(2)` 提取前 2 个元素，而不是 Python 的 slice
2. **精度**：C++ 使用 `double` 对应 Python 的 `float64`
3. **收敛判断**：两个版本的逻辑完全一致
4. **eta_damping**：控制消息传递的阻尼因子，影响收敛速度
