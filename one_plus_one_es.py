import numpy as np
import matplotlib.pyplot as plt

# Định nghĩa hàm mục tiêu (Fitness Function)
# Giả sử: Tối thiểu hóa hàm Sphere (f(x) = sum(x^2))
def objective_function(solution):
    """Hàm mục tiêu (giả định là bài toán tối thiểu hóa)."""
    # fitness(solution) là hàm đánh giá độ phù hợp của một giải pháp
    return np.sum(solution**2)

def one_plus_one_es(
    dim,
    bounds,
    mu=0.0,
    sigma=0.5,
    max_iterations=1000
):

    # Lấy giới hạn
    lower_bound, upper_bound = bounds

    # 1. Khởi tạo parent (cá thể cha) với một giải pháp ngẫu nhiên
    parent = np.random.uniform(lower_bound, upper_bound, dim)
    parent_fitness = objective_function(parent)

    fitness_history = [parent_fitness]

    print(f"Khởi tạo: Fitness ban đầu = {parent_fitness:.4f}")

    # Vòng lặp chính
    for i in range(1, max_iterations + 1):

        # 2. Đột biến (Mutation): Tạo offspring (cá thể con)
        # Thêm nhiễu ngẫu nhiên Normal(mu, sigma)
        mutation_vector = np.random.normal(mu, sigma, dim)
        offspring = parent + mutation_vector

        # Đảm bảo offspring nằm trong giới hạn (Tùy chọn)
        offspring = np.clip(offspring, lower_bound, upper_bound)

        # Tính fitness của offspring
        offspring_fitness = objective_function(offspring)

        # 3. Chọn lọc (Selection)
        # Nếu offspring có fitness tốt hơn (tối thiểu hóa: nhỏ hơn) so với parent
        if offspring_fitness < parent_fitness:
            # 4. Cập nhật: offspring thay thế parent
            parent = offspring
            parent_fitness = offspring_fitness

        # Fitness luôn giảm hoặc bằng, không bao giờ tăng (do cơ chế chọn lọc đơn giản)
        fitness_history.append(parent_fitness)

    # 5. Trả về parent như là giải pháp tốt nhất tìm thấy
    return parent, parent_fitness, fitness_history


# --- Ví dụ sử dụng với Input đã định nghĩa ---
if __name__ == "__main__":

    # Thiết lập tham số (Dựa trên ví dụ đã thảo luận)
    DIMENSION = 3          # Số chiều của bài toán
    BOUNDS = (-10.0, 10.0) # Giới hạn của các biến quyết định
    MAX_ITER = 1000        # Số lần lặp tối đa
    SIGMA_START = 1.0      # Độ mạnh đột biến ban đầu

    # Chạy thuật toán
    best_solution, best_fitness, history = one_plus_one_es(
        dim=DIMENSION,
        bounds=BOUNDS,
        sigma=SIGMA_START,
        max_iterations=MAX_ITER
    )

    print("\n--- Kết quả (1+1)-ES ---")
    print(f"Số lần lặp tối đa: {MAX_ITER}")
    print(f"Fitness tốt nhất tìm thấy: {best_fitness:.6f}")
    print(f"Giải pháp tốt nhất (X): {best_solution}")

    # Vẽ đồ thị kết quả
    plt.figure(figsize=(10, 6))
    plt.plot(history, label='Best Fitness Value', color='blue')
    plt.title('(1+1)-Evolution Strategy Convergence Plot (f(x) = sum(x^2))')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value (Fitness)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()