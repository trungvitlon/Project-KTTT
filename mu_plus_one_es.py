import numpy as np
import copy
import matplotlib.pyplot as plt
from typing import List, Tuple

# Định nghĩa hàm mục tiêu (Fitness Function). Giả định bài toán TỐI THIỂU HÓA.
def objective_function(x: np.ndarray) -> float:
    """Hàm tổng bình phương: f(x) = sum(x^2)."""
    return np.sum(x**2)

class Individual:
    """Cấu trúc đại diện cho một cá thể."""
    def __init__(self, num_dims: int, initial_step_size: float):
        # Biến Đối tượng (Object Variables)
        self.x = np.random.uniform(-5.0, 5.0, num_dims) 
        # Tham số Chiến lược (Strategy Parameters: mutation step sizes)
        self.sigma = np.array([initial_step_size] * num_dims)
        self.fitness = objective_function(self.x)

def mu_plus_one_es_simple(mu: int, num_dimensions: int, max_generations: int, initial_step_size: float = 1.0) -> Tuple[Individual, List[float]]:
    
    # ----------------------------------------------------
    # BƯỚC KHỞI TẠO
    # ----------------------------------------------------
    
    # Khởi tạo quần thể ban đầu gồm μ cá thể
    population = [Individual(num_dimensions, initial_step_size) for _ in range(mu)]
    
    # Ghi nhận cá thể tốt nhất ban đầu
    best_individual = min(population, key=lambda ind: ind.fitness)
    fitness_history: List[float] = [best_individual.fitness]
    
    # Định nghĩa các hằng số thích nghi cho đột biến log-normal
    tau = 1.0 / np.sqrt(2 * num_dimensions) 
    tau_prime = 1.0 / np.sqrt(2 * np.sqrt(num_dimensions))

    # ----------------------------------------------------
    # VÒNG LẶP CHÍNH
    # ----------------------------------------------------
    
    for gen in range(max_generations):
        
        # 1. Chọn Cá thể Cha mẹ (P)
        # Chọn ngẫu nhiên một cá thể từ quần thể
        parent_index = np.random.randint(mu)
        parent = population[parent_index]
        
        # 2. Tạo Cá thể Con (O) bằng cách Đột biến Cha mẹ
        offspring = copy.deepcopy(parent)
        
        # Đột biến Tham số Chiến lược (log-normal distribution)
        global_rand = np.random.normal(0, 1) 
        for i in range(num_dimensions):
            # Nhân tham số chiến lược với yếu tố ngẫu nhiên từ phân phối log-normal
            lognormal_factor = np.exp(tau_prime * global_rand + tau * np.random.normal(0, 1))
            offspring.sigma[i] *= lognormal_factor
            
        # Đột biến Biến Đối tượng (normal distribution)
        for i in range(num_dimensions):
            # Cộng giá trị ngẫu nhiên từ phân phối chuẩn (SD = strategy parameter)
            sigma_i = offspring.sigma[i]
            offspring.x[i] += np.random.normal(0, sigma_i)
            
        # Đánh giá Fitness của Cá thể con
        offspring.fitness = objective_function(offspring.x)
        
        # 3. Chọn lọc Cá thể Sống sót
        # Nếu con có fitness tốt hơn (hoặc bằng) so với cha mẹ (cho bài toán tối thiểu hóa)
        if offspring.fitness <= parent.fitness: 
            # Thay thế cha mẹ bằng cá thể con trong quần thể
            population[parent_index] = offspring
            
            # Cập nhật cá thể tốt nhất
            if offspring.fitness < best_individual.fitness:
                best_individual = offspring
        
        # Ghi lại fitness tốt nhất sau mỗi thế hệ
        fitness_history.append(best_individual.fitness)

    # Trả về cá thể tốt nhất và lịch sử fitness
    return best_individual, fitness_history


MU_SIZE = 10         # μ: Số lượng cá thể cha mẹ
DIMS = 5             # Số chiều của bài toán
MAX_GEN = 500        # Số lần lặp tối đa
INIT_STEP = 1.0      # Kích thước bước ban đầu

# Chạy thuật toán
best, history = mu_plus_one_es_simple(
    mu=MU_SIZE,
    num_dimensions=DIMS,
    max_generations=MAX_GEN,
    initial_step_size=INIT_STEP
)

print(f"Thuật toán (μ={MU_SIZE}+1)-ES kết thúc sau {MAX_GEN} thế hệ.")
print(f"Giá trị Fitness tốt nhất: {best.fitness:.6e}")
print(f"Giải pháp tốt nhất (Object Variables): {best.x}")


# Vẽ đồ thị hội tụ
plt.figure(figsize=(10, 6))
# Bỏ qua phần tử đầu tiên trong history vì nó là giá trị khởi tạo trước vòng lặp
plt.plot(range(1, MAX_GEN + 1), history[1:], label='Fitness Tốt nhất')
plt.title(f'Sự Hội Tụ của ({MU_SIZE}+1)-Evolution Strategy')
plt.xlabel('Thế hệ')
plt.ylabel('Giá trị Fitness (Tối thiểu hóa)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()