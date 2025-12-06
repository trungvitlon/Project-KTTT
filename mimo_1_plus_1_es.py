import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal, uniform

# ================== 1. THAM SỐ HỆ THỐNG ==================
Mt = 2      # Số lượng AP phát
Nt = 16     # Số anten trên mỗi AP
U = 5       # Số lượng người dùng (UEs)
Q = 1       # Số luồng cảm biến
S = U + Q   # Tổng số luồng
DIMENSION = Mt * Nt * S * 2 # Tổng số biến thực (phần thực và ảo của fms)

MAX_POWER = 1.0 # P_m = 1 Watt
MIN_SINR_THRES = np.full(U, 1.0) # Ngưỡng SINR tối thiểu (gamma_u)

# Tham số (1+1)-ES
ES_SIGMA_INIT = 0.1 
ES_MAX_ITERS = 5000 
ES_MU = 0.0          
PENALTY_FACTOR = 1000 # Hệ số phạt cho việc vi phạm ràng buộc

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

# ================== 5. (1+1)-EVOLUTION STRATEGY ==================
def initialize_parent(dimension, max_power):
    amplitude_limit = np.sqrt(max_power / (dimension // 2))
    initial_solution = uniform(-amplitude_limit, amplitude_limit, dimension)
    return initial_solution

def mutation(parent, sigma, mu):
    perturbation = normal(mu, sigma, parent.shape)
    offspring = parent + perturbation
    return offspring

def evolution_strategy_1_plus_1(dimension, max_iters, sigma_init, mu, system_params):
    P_MAX = system_params['P_max'] # Access P_max from system_params
    parent = initialize_parent(dimension, P_MAX)
    parent_fitness, parent_snr, parent_penalty = calculate_fitness(
        parent, **system_params
    )
    best_solution = parent
    best_fitness = parent_fitness
    current_sigma = sigma_init
    fitness_history = [parent_fitness]
    snr_history = [parent_snr]
    penalty_history = [parent_penalty]
    print(f"Bắt đầu (1+1)-ES. Fitness khởi tạo: {parent_fitness:.4e}")
    for i in range(max_iters):
        offspring = mutation(parent, current_sigma, mu)
        offspring_fitness, offspring_snr, offspring_penalty = calculate_fitness(
            offspring, **system_params
        )
        if offspring_fitness > parent_fitness:
            parent = offspring
            parent_fitness = offspring_fitness
            if parent_fitness > best_fitness:
                best_fitness = parent_fitness
                best_solution = parent
        fitness_history.append(parent_fitness)
        snr_history.append(offspring_snr)
        penalty_history.append(offspring_penalty)
        if (i + 1) % 500 == 0:
            print(f"Lặp {i + 1}/{max_iters}: Fitness tốt nhất = {best_fitness:.4e}")
    print(f"\nOptimization finished after {max_iters} iterations.")
    print(f"Final Best Fitness (SNR - Penalty): {best_fitness:.4e}")
    # Trả về vector phức tạp và lịch sử hội tụ
    return vector_to_complex_beamforming(best_solution, Mt, Nt, S), fitness_history, snr_history, penalty_history

# ================== 6. MAIN & ĐỒ THỊ ==================
def main():
    H_U_PL = generate_placeholder_channels(U, Mt, Nt)
    A_RESP_PL = generate_array_response(Mt, Nt)
    system_parameters = {
        'Mt': Mt,
        'Nt': Nt,
        'U': U,
        'Q': Q,
        'P_max': MAX_POWER,
        'gamma_u': MIN_SINR_THRES,
        'H_U': H_U_PL,
        'A_resp': A_RESP_PL,
        'sensing_params': SENSING_PARAMS
    }
    final_beamforming_solution, fitness_history, snr_history, penalty_history = evolution_strategy_1_plus_1(
        DIMENSION, ES_MAX_ITERS, ES_SIGMA_INIT, ES_MU, system_parameters
    )
    print("\n===============================================")
    print("KẾT QUẢ ĐỊNH DẠNG CHÙM TIA TỐI ƯU")
    print("===============================================")
    if final_beamforming_solution is not None:
        print(f"Vector f_0,0 (AP 0, Luồng 0, {Nt}x1):")
        print(final_beamforming_solution[0][0][:5])
        sensing_stream_index = U
        if final_beamforming_solution[0][sensing_stream_index] is not None:
            print(f"\nVector f_0,Q (AP 0, Luồng Cảm biến {Q}, {Nt}x1):")
            print(final_beamforming_solution[0][sensing_stream_index][:5])
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