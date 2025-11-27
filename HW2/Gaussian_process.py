import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import seaborn as sns
from scipy.stats import loguniform

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class GaussianProcess:
    def __init__(self, kernel_type='SE'):
        self.kernel_type = kernel_type
        self.params = None
        self.X_train = None
        self.y_train = None
        self.optimization_history = []
        
    def se_kernel(self, X1, X2, l, sigma_f, sigma_n):
        """平方指数核函数"""
        if X1.ndim == 1:
            X1 = X1.reshape(-1, 1)
        if X2.ndim == 1:
            X2 = X2.reshape(-1, 1)
            
        sq_dist = cdist(X1, X2, metric='sqeuclidean')
        K = sigma_f**2 * np.exp(-0.5 * sq_dist / l**2)
        
        # 添加噪声项（只在训练数据上）
        if X1.shape == X2.shape and np.array_equal(X1, X2):
            K += sigma_n**2 * np.eye(X1.shape[0])
            
        return K
    
    def matern_kernel(self, X1, X2, l, sigma_f, sigma_n):
        """Matern 3/2 核函数"""
        if X1.ndim == 1:
            X1 = X1.reshape(-1, 1)
        if X2.ndim == 1:
            X2 = X2.reshape(-1, 1)
            
        dist = cdist(X1, X2, metric='euclidean')
        r = dist / l
        K = sigma_f**2 * (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
        
        # 添加噪声项
        if X1.shape == X2.shape and np.array_equal(X1, X2):
            K += sigma_n**2 * np.eye(X1.shape[0])
            
        return K
    
    def kernel(self, X1, X2, params):
        """统一的核函数接口"""
        l, sigma_f, sigma_n = params
        if self.kernel_type == 'SE':
            return self.se_kernel(X1, X2, l, sigma_f, sigma_n)
        elif self.kernel_type == 'Matern':
            return self.matern_kernel(X1, X2, l, sigma_f, sigma_n)
        else:
            raise ValueError("不支持的核函数类型")
    
    def stable_cholesky(self, K, max_jitter=1e-6):
        """数值稳定的Cholesky分解"""
        jitter = 1e-12
        n = K.shape[0]
        
        for _ in range(20):
            try:
                L = np.linalg.cholesky(K + jitter * np.eye(n))
                return L, jitter
            except np.linalg.LinAlgError:
                jitter *= 10
                if jitter > max_jitter:
                    break
        
        # 如果仍然失败，使用SVD近似
        U, s, Vt = np.linalg.svd(K)
        s = np.maximum(s, 1e-12)  # 确保特征值为正
        L = U @ np.diag(np.sqrt(s))
        return L, max_jitter
    
    def log_marginal_likelihood(self, params):
        """对数边际似然（数值稳定版本）"""
        try:
            K = self.kernel(self.X_train, self.X_train, params)
            L, jitter_used = self.stable_cholesky(K)
            
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
            
            # 计算对数边际似然
            log_likelihood = -0.5 * self.y_train.T @ alpha
            log_likelihood -= np.sum(np.log(np.diag(L)))
            log_likelihood -= self.X_train.shape[0] / 2 * np.log(2 * np.pi)
            
            return log_likelihood
            
        except Exception as e:
            return -1e10  # 返回一个很小的值
    
    def negative_log_likelihood(self, params):
        """负对数边际似然"""
        return -self.log_marginal_likelihood(params)
    
    def generate_initial_guesses(self, n_guesses=20):
        """生成多样化的初始猜测点"""
        guesses = []
        
        # 在关键区域采样
        for i in range(n_guesses):
            if i < n_guesses // 3:
                # 小长度尺度区域
                l = loguniform.rvs(0.1, 1.0)
                sigma_f = loguniform.rvs(0.5, 2.0)
                sigma_n = loguniform.rvs(0.01, 0.3)
            elif i < 2 * n_guesses // 3:
                # 中等长度尺度区域
                l = loguniform.rvs(0.5, 3.0)
                sigma_f = loguniform.rvs(0.5, 2.0)
                sigma_n = loguniform.rvs(0.01, 0.3)
            else:
                # 大长度尺度区域
                l = loguniform.rvs(1.0, 10.0)
                sigma_f = loguniform.rvs(0.5, 2.0)
                sigma_n = loguniform.rvs(0.01, 0.3)
            
            guesses.append([l, sigma_f, sigma_n])
        
        return np.array(guesses)
    
    def fit_global(self, X, y, n_restarts=25, verbose=True):
        """全局优化拟合（多起点优化）"""
        self.X_train = X.reshape(-1, 1)
        self.y_train = y
        self.optimization_history = []
        
        # 生成多样化的初始点
        initial_guesses = self.generate_initial_guesses(n_restarts)
        
        best_params = None
        best_log_likelihood = -np.inf
        
        bounds = [(1e-5, 20), (1e-5, 20), (1e-5, 2)]
        
        if verbose:
            print(f"开始全局优化，使用 {n_restarts} 个初始点...")
        
        for i, initial_guess in enumerate(initial_guesses):
            try:
                result = minimize(
                    self.negative_log_likelihood,
                    initial_guess,
                    bounds=bounds,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-8}
                )
                
                current_log_likelihood = -result.fun
                
                self.optimization_history.append({
                    'initial': initial_guess,
                    'optimal': result.x,
                    'nll': result.fun,
                    'log_likelihood': current_log_likelihood,
                    'success': result.success
                })
                
                if current_log_likelihood > best_log_likelihood and result.success:
                    best_log_likelihood = current_log_likelihood
                    best_params = result.x
                    
                if verbose and (i+1) % 5 == 0:
                    print(f"完成 {i+1}/{n_restarts} 次优化，当前最佳对数似然: {best_log_likelihood:.4f}")
                    
            except Exception as e:
                if verbose:
                    print(f"优化 {i+1} 失败: {e}")
                continue
        
        if best_params is None:
            # 如果所有优化都失败，使用负对数似然最小的那个
            best_run = min(self.optimization_history, key=lambda x: x['nll'])
            best_params = best_run['optimal']
            best_log_likelihood = best_run['log_likelihood']
            print("警告：使用备选参数")
        
        self.params = best_params
        
        if verbose:
            print(f"\n全局优化完成!")
            print(f"最佳对数似然: {best_log_likelihood:.4f}")
            print(f"最优参数: l={best_params[0]:.6f}, σ_f={best_params[1]:.6f}, σ_n={best_params[2]:.6f}")
        
        return best_params
    
    def fit(self, X, y, method='global', n_restarts=25):
        """训练高斯过程（兼容原有接口）"""
        if method == 'global':
            return self.fit_global(X, y, n_restarts)
        else:
            # 保持原有的单起点优化
            self.X_train = X.reshape(-1, 1)
            self.y_train = y
            
            initial_params = [1.0, 1.0, 0.1]
            bounds = [(1e-5, 10), (1e-5, 10), (1e-5, 1)]
            
            result = minimize(self.negative_log_likelihood, initial_params, 
                             bounds=bounds, method='L-BFGS-B')
            
            self.params = result.x
            return self.params
    
    def predict(self, X_test):
        """预测"""
        K = self.kernel(self.X_train, self.X_train, self.params)
        K_s = self.kernel(self.X_train, X_test, self.params)
        K_ss = self.kernel(X_test, X_test, self.params)
        
        # 使用稳定的Cholesky分解
        L, _ = self.stable_cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
        
        # 预测均值
        mu = K_s.T @ alpha
        
        # 预测方差（数值稳定版本）
        v = np.linalg.solve(L, K_s)
        cov = K_ss - v.T @ v
        var = np.diag(cov)
        var = np.maximum(var, 0)  # 确保方差非负
        
        return mu.flatten(), var
    
    def analyze_optimization(self):
        """分析优化结果"""
        if not self.optimization_history:
            print("没有优化历史数据")
            return
        
        nll_values = [run['nll'] for run in self.optimization_history]
        success_count = sum(run['success'] for run in self.optimization_history)
        
        print(f"\n=== 优化结果分析 ===")
        print(f"总运行次数: {len(self.optimization_history)}")
        print(f"成功次数: {success_count}")
        print(f"成功率: {success_count/len(self.optimization_history)*100:.1f}%")
        print(f"负对数似然范围: {min(nll_values):.4f} ~ {max(nll_values):.4f}")
        
        # 分析不同的局部解
        unique_solutions = {}
        for run in self.optimization_history:
            if run['success']:
                key = tuple(np.round(run['optimal'], 2))
                if key not in unique_solutions:
                    unique_solutions[key] = []
                unique_solutions[key].append(run['log_likelihood'])
        
        print(f"\n找到 {len(unique_solutions)} 个不同的局部解")
        for i, (params, lls) in enumerate(list(unique_solutions.items())[:3]):
            print(f"解 {i+1}: l={params[0]:.2f}, σ_f={params[1]:.2f}, σ_n={params[2]:.2f}, "
                  f"平均对数似然={np.mean(lls):.4f}")

def generate_data():
    """生成训练数据"""
    np.random.seed(42)
    
    # 生成20个随机点
    X_train = np.random.uniform(-8, 8, 20)
    
    # 使用固定参数的高斯过程生成数据
    gp_true = GaussianProcess(kernel_type='SE')
    params_true = [1.0, 1.0, 0.1]  # l, sigma_f, sigma_n
    
    # 计算协方差矩阵
    K = gp_true.se_kernel(X_train, X_train, *params_true)
    
    # 从多元高斯分布采样
    y_train = np.random.multivariate_normal(np.zeros_like(X_train), K)
    
    return X_train, y_train, params_true

def plot_results(X_train, y_train, X_test, mu, sigma, title, params):
    """绘制结果"""
    plt.figure(figsize=(10, 6))
    
    # 绘制训练数据
    plt.scatter(X_train, y_train, c='red', marker='x', label='训练数据', zorder=5)
    
    # 绘制预测均值
    plt.plot(X_test, mu, 'b-', label='预测均值')
    
    # 绘制不确定性区域
    plt.fill_between(X_test, mu - 2*sigma, mu + 2*sigma, 
                    alpha=0.3, color='blue', label='2σ不确定性')
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'{title}\n参数: l={params[0]:.3f}, σ_f={params[1]:.3f}, σ_n={params[2]:.6f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{title}.png', dpi=500)
    plt.close()

def plot_contour(X_train, y_train, kernel_type, optimal_params=None, fixed_param_idx=None):
    """绘制对数似然等高线图（修复版本）"""
    gp = GaussianProcess(kernel_type=kernel_type)
    gp.X_train = X_train.reshape(-1, 1)
    gp.y_train = y_train
    
    if optimal_params is None:
        optimal_params = [1.0, 1.0, 0.1]
    
    # 生成参数网格 - 修复：确保每个维度至少有2个点
    if fixed_param_idx == 0:  # 固定l
        l_values = np.array([optimal_params[0]])  # 单个值，但我们会用其他两个参数创建网格
        sigma_f_values = np.linspace(0.1, 2.0, 30)
        sigma_n_values = np.linspace(0.01, 0.5, 30)
        fixed_param_name = 'l'
        fixed_value = optimal_params[0]
        
        # 创建网格
        X, Y = np.meshgrid(sigma_f_values, sigma_n_values)
        Z = np.zeros_like(X)
        
        # 计算每个点的对数似然
        for i in range(len(sigma_f_values)):
            for j in range(len(sigma_n_values)):
                params = [fixed_value, sigma_f_values[i], sigma_n_values[j]]
                Z[j, i] = gp.log_marginal_likelihood(params)
                
    elif fixed_param_idx == 1:  # 固定sigma_f
        l_values = np.linspace(0.1, 3.0, 30)
        sigma_f_values = np.array([optimal_params[1]])
        sigma_n_values = np.linspace(0.01, 0.5, 30)
        fixed_param_name = 'σ_f'
        fixed_value = optimal_params[1]
        
        # 创建网格
        X, Y = np.meshgrid(l_values, sigma_n_values)
        Z = np.zeros_like(X)
        
        # 计算每个点的对数似然
        for i in range(len(l_values)):
            for j in range(len(sigma_n_values)):
                params = [l_values[i], fixed_value, sigma_n_values[j]]
                Z[j, i] = gp.log_marginal_likelihood(params)
                
    else:  # 固定sigma_n
        l_values = np.linspace(0.1, 3.0, 30)
        sigma_f_values = np.linspace(0.1, 2.0, 30)
        sigma_n_values = np.array([optimal_params[2]])
        fixed_param_name = 'σ_n'
        fixed_value = optimal_params[2]
        
        # 创建网格
        X, Y = np.meshgrid(l_values, sigma_f_values)
        Z = np.zeros_like(X)
        
        # 计算每个点的对数似然
        for i in range(len(l_values)):
            for j in range(len(sigma_f_values)):
                params = [l_values[i], sigma_f_values[j], fixed_value]
                Z[j, i] = gp.log_marginal_likelihood(params)
    
    # 绘制等高线图
    plt.figure(figsize=(10, 8))
    
    if fixed_param_idx == 0:
        contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.xlabel('σ_f')
        plt.ylabel('σ_n')
    elif fixed_param_idx == 1:
        contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.xlabel('l')
        plt.ylabel('σ_n')
    else:
        contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.xlabel('l')
        plt.ylabel('σ_f')
    
    # 标记最优解位置
    if fixed_param_idx == 0:
        plt.plot(optimal_params[1], optimal_params[2], 'ro', markersize=8, label='最优解')
    elif fixed_param_idx == 1:
        plt.plot(optimal_params[0], optimal_params[2], 'ro', markersize=8, label='最优解')
    else:
        plt.plot(optimal_params[0], optimal_params[1], 'ro', markersize=8, label='最优解')
    
    plt.colorbar(contour, label='对数边际似然')
    plt.legend()
    kernel_name = '平方指数核' if kernel_type == 'SE' else 'Matern核'
    plt.title(f'{kernel_name}对数边际似然等高线图 (固定{fixed_param_name}={fixed_value:.4f})')
    plt.savefig(f'{kernel_name}对数边际似然等高线图 (固定{fixed_param_name}={fixed_value:.4f}).png', dpi=500)
    plt.close()

def main():
    # 1. 生成数据
    print("生成训练数据...")
    X_train, y_train, true_params = generate_data()
    
    # 测试点
    X_test = np.linspace(-8, 8, 200)
    
    # 2. 使用不同参数进行回归
    print("\n=== 使用不同参数进行回归 ===")
    
    param_sets = [
        ([1.0, 1.0, 0.1], "参数集1"),
        ([0.3, 1.08, 5e-5], "参数集2"), 
        ([3.0, 1.16, 0.89], "参数集3")
    ]
    
    for params, name in param_sets:
        print(f"\n使用{name}: {params}")
        gp = GaussianProcess(kernel_type='SE')
        gp.X_train = X_train.reshape(-1, 1)
        gp.y_train = y_train
        gp.params = params
        
        mu, var = gp.predict(X_test)
        sigma = np.sqrt(var)
        
        plot_results(X_train, y_train, X_test, mu, sigma, 
                    f"高斯过程回归 - {name}", params)
    
    # 3. 贝叶斯模型选择 - 平方指数核（使用全局优化）
    print("\n=== 贝叶斯模型选择 - 平方指数核 ===")
    gp_se = GaussianProcess(kernel_type='SE')
    optimal_params_se = gp_se.fit_global(X_train, y_train, n_restarts=100)
    gp_se.analyze_optimization()
    
    mu_se, var_se = gp_se.predict(X_test)
    sigma_se = np.sqrt(var_se)
    plot_results(X_train, y_train, X_test, mu_se, sigma_se, 
                "贝叶斯模型选择 - 平方指数核", optimal_params_se)
    
    # 绘制等高线图
    print("绘制平方指数核等高线图...")
    plot_contour(X_train, y_train, 'SE', optimal_params_se, fixed_param_idx=2)
    plot_contour(X_train, y_train, 'SE', optimal_params_se, fixed_param_idx=0)
    plot_contour(X_train, y_train, 'SE', optimal_params_se, fixed_param_idx=1)
    
    # 4. 贝叶斯模型选择 - Matern核（使用全局优化）
    print("\n=== 贝叶斯模型选择 - Matern核 ===")
    gp_matern = GaussianProcess(kernel_type='Matern')
    optimal_params_matern = gp_matern.fit_global(X_train, y_train, n_restarts=100)
    gp_matern.analyze_optimization()
    
    mu_matern, var_matern = gp_matern.predict(X_test)
    sigma_matern = np.sqrt(var_matern)
    plot_results(X_train, y_train, X_test, mu_matern, sigma_matern, 
                "贝叶斯模型选择 - Matern核", optimal_params_matern)
    
    # 绘制等高线图
    print("绘制Matern核等高线图...")
    plot_contour(X_train, y_train, 'Matern', optimal_params_matern, fixed_param_idx=2)
    plot_contour(X_train, y_train, 'Matern', optimal_params_matern, fixed_param_idx=0)
    plot_contour(X_train, y_train, 'Matern', optimal_params_matern, fixed_param_idx=1)
    
    # 5. 比较不同核函数的结果
    print("\n=== 结果比较 ===")
    print(f"真实参数: l={true_params[0]}, σ_f={true_params[1]}, σ_n={true_params[2]}")
    print(f"SE核最优参数: l={optimal_params_se[0]:.6f}, σ_f={optimal_params_se[1]:.6f}, σ_n={optimal_params_se[2]:.6f}")
    print(f"Matern核最优参数: l={optimal_params_matern[0]:.6f}, σ_f={optimal_params_matern[1]:.6f}, σ_n={optimal_params_matern[2]:.6f}")

if __name__ == "__main__":
    main()