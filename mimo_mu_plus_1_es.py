import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal, uniform, randint

# ================== 1. THAM SỐ HỆ THỐNG ==================
Mt = 2      # Số lượng AP phát
Nt = 16     # Số anten trên mỗi AP
U = 5       # Số lượng người dùng (UEs)
Q = 1       # Số luồng cảm biến
S = U + Q   # Tổng số luồng
DIMENSION = Mt * Nt * S * 2 # Tổng số biến thực (phần thực và ảo của fms)

MAX_POWER = 1.0 # P_m = 1 Watt
MIN_SINR_THRES = np.full(U, 1.0) # Ngưỡng SINR tối thiểu (gamma_u)
PENALTY_FACTOR = 1000 # Hệ số phạt cho việc vi phạm ràng buộc

# Tham số (μ+1)-ES
MU = 10 # Number of individuals
ES_SIGMA_INIT = 0.1
ES_MAX_ITERS = 5000
ES_MU = 0.0
LEARNING_RATE_GLOBAL = 1.0 / np.sqrt(2 * np.sqrt(DIMENSION))
LEARNING_RATE_LOCAL = 1.0 / np.sqrt(2 * DIMENSION)

# ================== 2. HÀM GIẢ LẬP KÊNH & ĐÁP ỨNG ==================
def generate_placeholder_channels(U, Mt, Nt):
    H_U = {}
    for u in range(U):
        H_U[u] = (normal(0, 1, Mt * Nt) + 1j * normal(0, 1, Mt * Nt)) * np.sqrt(0.5)
    return H_U

def generate_array_response(Mt, Nt):
    A_response = {}
    for m in range(Mt):
        A_response[m] = np.ones((Nt, 1)) + 1j * np.zeros((Nt, 1))
    return A_response

SENSING_PARAMS = {
    'Zeta_sq': 0.01,
    'Sigma_sq_comm': np.full(U, 1e-6),
    'Sigma_sq_sensing': 1.0
}

# ================== 3. CHUYỂN ĐỔI BIẾN ==================
def vector_to_complex_beamforming(vector_real, Mt, Nt, S):
    f_ms = {}
    complex_vector = vector_real[::2] + 1j * vector_real[1::2]
    for m in range(Mt):
        f_ms[m] = {}
        for s in range(S):
            start_idx = (m * S + s) * Nt
            end_idx = start_idx + Nt
            f_ms[m][s] = complex_vector[start_idx:end_idx].reshape(Nt, 1)
    return f_ms

# ================== 4. HÀM RÀNG BUỘC & MỤC TIÊU ==================
def calculate_power_constraint(f_ms, Mt, S, P_max):
    total_penalty = 0.0
    for m in range(Mt):
        ap_power = 0.0
        for s in range(S):
            ap_power += np.sum(np.abs(f_ms[m][s]) ** 2)
        if ap_power > P_max:
            penalty = (ap_power - P_max)
            total_penalty += penalty * PENALTY_FACTOR
    return total_penalty

def calculate_comm_SINR(f_ms, u, H_U, Mt, Nt, U, Q, sigma_sq_u, gamma_u):
    h_mu = {} 
    h_u = H_U[u]
    for m in range(Mt):
        start_idx = m * Nt
        end_idx = (m + 1) * Nt
        h_mu[m] = h_u[start_idx:end_idx].reshape(Nt, 1)
    DS_sum = sum(h_mu[m].conj().T @ f_ms[m][u] for m in range(Mt))
    DS_power = np.abs(DS_sum) ** 2
    MUI_power = sum(
        np.abs(sum(h_mu[m].conj().T @ f_ms[m][u_prime] for m in range(Mt))) ** 2
        for u_prime in range(U) if u_prime != u
    )
    SI_power = sum(
        np.abs(sum(h_mu[m].conj().T @ f_ms[m][q] for m in range(Mt))) ** 2
        for q in range(U, U+Q)
    )
    Noise_power = sigma_sq_u
    SINR_u = DS_power / (MUI_power + SI_power + Noise_power)
    penalty = 0.0
    if SINR_u < gamma_u:
        penalty = (gamma_u - SINR_u) * PENALTY_FACTOR
    return float(SINR_u), float(penalty)

def calculate_sensing_SNR(f_ms, Mt, S, A_resp, sensing_params):
    varsigma_sq = sensing_params['Sigma_sq_sensing']
    Mr = Mt
    denominator = Mr * varsigma_sq
    numerator = 0.0
    zeta_sq = sensing_params['Zeta_sq']
    for m_t in range(Mt):
        F_mt = np.hstack([f_ms[m_t][s] for s in range(S)])
        a_mt = A_resp[m_t]
        inner_term = a_mt.conj().T @ F_mt
        norm_sq = np.sum(np.abs(inner_term) ** 2)
        numerator += zeta_sq * Mr * norm_sq
    return float(numerator / denominator) if denominator != 0 else 0.0

def calculate_fitness(vector_real, Mt, Nt, U, Q, P_max, gamma_u, H_U, A_resp, sensing_params):
    S = U + Q
    f_ms = vector_to_complex_beamforming(vector_real, Mt, Nt, S)
    total_penalty = 0.0
    total_penalty += calculate_power_constraint(f_ms, Mt, S, P_max)
    for u in range(U):
        sigma_sq_u = sensing_params['Sigma_sq_comm'][u]
        gamma_u_thres = gamma_u[u]
        _, sinr_penalty = calculate_comm_SINR(
            f_ms, u, H_U, Mt, Nt, U, Q, sigma_sq_u, gamma_u_thres
        )
        total_penalty += sinr_penalty
    sensing_snr = calculate_sensing_SNR(f_ms, Mt, S, A_resp, sensing_params)
    fitness = sensing_snr - total_penalty
    return float(fitness), float(sensing_snr), float(total_penalty)

# ================== 5. (μ+1)-EVOLUTION STRATEGY ==================
class Individual:
    def __init__(self, x, sigma, fitness=None, snr=None, penalty=None):
        self.x = x
        self.sigma = sigma
        self.fitness = fitness
        self.snr = snr
        self.penalty = penalty

def initialize_population(mu, dimension, sigma_init, max_power):
    population = []
    amplitude_limit = np.sqrt(max_power / (dimension // 2))
    for _ in range(mu):
        x = uniform(-amplitude_limit, amplitude_limit, dimension)
        sigma = np.full(dimension, sigma_init)
        individual = Individual(x, sigma)
        population.append(individual)
    return population

def mutate_individual(parent):
    dimension = len(parent.x)
    N_global = normal(0, 1)
    new_sigma = parent.sigma * np.exp(
        LEARNING_RATE_GLOBAL * N_global +
        LEARNING_RATE_LOCAL * normal(0, 1, dimension)
    )
    new_sigma = np.clip(new_sigma, 1e-6, None)
    new_x = parent.x + new_sigma * normal(ES_MU, 1, dimension)
    return Individual(new_x, new_sigma)

def evolution_strategy_mu_plus_1(dimension, mu, max_iters, sigma_init, system_params):
    P_MAX = system_params['P_max'] # Access P_max from system_params
    population = initialize_population(mu, dimension, sigma_init, P_MAX)
    fitness_history = []
    snr_history = []
    penalty_history = []
    for individual in population:
        individual.fitness, individual.snr, individual.penalty = calculate_fitness(individual.x, **system_params)
    population.sort(key=lambda ind: ind.fitness, reverse=True)
    best_individual = population[0]
    fitness_history.append(best_individual.fitness)
    snr_history.append(best_individual.snr)
    penalty_history.append(best_individual.penalty)
    print(f"Bắt đầu (μ+1)-ES (μ={mu}). Fitness tốt nhất khởi tạo: {best_individual.fitness:.4e}")
    for i in range(max_iters):
        parent_index = randint(mu)
        parent = population[parent_index]
        offspring = mutate_individual(parent)
        offspring.fitness, offspring.snr, offspring.penalty = calculate_fitness(offspring.x, **system_params)
        combined_population = population + [offspring]
        combined_population.sort(key=lambda ind: ind.fitness, reverse=True)
        population = combined_population[:mu]
        current_best = population[0]
        if current_best.fitness > best_individual.fitness:
            best_individual = current_best
        fitness_history.append(best_individual.fitness)
        snr_history.append(best_individual.snr)
        penalty_history.append(best_individual.penalty)
        if (i + 1) % 500 == 0:
            print(f"Lặp {i + 1}/{max_iters}: Fitness tốt nhất = {best_individual.fitness:.4e}")
    print(f"\nOptimization finished after {max_iters} iterations.")
    print(f"Final Best Fitness: {best_individual.fitness:.4e}")
    return vector_to_complex_beamforming(best_individual.x, Mt, Nt, S), fitness_history, snr_history, penalty_history

def main():
    H_U_PL = generate_placeholder_channels(U, Mt, Nt)
    A_RESP_PL = generate_array_response(Mt, Nt)
    system_parameters = {
        'Mt': Mt,
        'Nt': Nt,
        'U': U,
        'Q': Q,
        'P_max': MAX_POWER,
        'gamma_u': MIN_SINR_THRES, # Changed key from 'MIN_SINR_THRES' to 'gamma_u'
        'H_U': H_U_PL,             # Changed key from 'H_U_PL' to 'H_U'
        'A_resp': A_RESP_PL,       # Changed key from 'A_RESP_PL' to 'A_resp'
        'sensing_params': SENSING_PARAMS # Changed key from 'SENSING_PARAMS' to 'sensing_params'
    }
    final_beamforming_solution, fitness_history, snr_history, penalty_history = evolution_strategy_mu_plus_1(
        DIMENSION, MU, ES_MAX_ITERS, ES_SIGMA_INIT, system_parameters # Changed ES_MAX_ITERS to MU
    )
    print("\n===============================================")
    print("KẾT QUẢ ĐỊNH DẠNG CHÙM TIA TỐI ƯU")
    print("===============================================")
    if final_beamforming_solution is not None:
        print(f"Số lượng vector f_ms tìm được (tại {Mt} AP và {S} luồng): {len(final_beamforming_solution)}")
        print(f"Vector f_0,0 (AP 0, Luồng 0, {Nt}x1) - 5 phần tử đầu:")
        print(final_beamforming_solution[0][0][:5])
    # Vẽ đồ thị hội tụ
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(fitness_history, label='Fitness', color='blue')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.subplot(3, 1, 2)
    plt.plot(snr_history, label='Sensing SNR', color='green')
    plt.ylabel('Sensing SNR')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.subplot(3, 1, 3)
    plt.plot(penalty_history, label='Penalty', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Penalty')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()