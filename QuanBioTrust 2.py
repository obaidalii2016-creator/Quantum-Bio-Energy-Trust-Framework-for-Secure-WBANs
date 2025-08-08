# quanbiotrust_advanced.py
# Advanced Quantum-Bio-Energy-Trust WBAN Framework
# Full 7-step protocol with detailed physics, trust, and quantum simulation
# Author: QuanBioTrust Research Team
# Date: 2025

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings("ignore")

# ==============================
# CONFIGURATION & PHYSICAL CONSTANTS
# ==============================

# Network scale
NUM_NODES = 35
NUM_ROUNDS = 120
TIME_STEP = 1.0  # seconds

# Energy parameters (Joules)
E_min = 0.5
E_max = 2.0
E_cap = 2.5
E_th = 0.2  # Saturation threshold
E_leak_rate = 1e-5  # Per timestep

# Harvesting constants
η_0 = 0.85  # Max efficiency
k_h_thermal = 0.012  # µW/cm²/K (thermoelectric)
k_h_piezo = 0.008   # µW/cm²/(m/s²) (piezoelectric)
A_mean = 1.8  # cm²
A_std = 0.3

# Communication
R_0 = 3.0  # Initial transmission range (m)
κ = 0.04   # Range decay over time
E_elec = 50e-6  # J/bit
ϵ_amp = 100e-6  # J/bit/m²
ϵ_m = 12e-6     # J/(m²/s²) for mobility
E_proc = 25e-6  # J per processing operation
E_qproc = 180e-6  # Quantum processing cost

# Quantum parameters
γ = 0.01  # Decoherence coefficient (per meter)
δ = 0.002  # Time-based decoherence
qubit_size = 5  # 2^5 = 32 bits
error_correction_factor = 3

# Trust parameters
⋋_s = 0.1  # Success rate constant
⋋_f = 0.2  # Failure rate constant
T_window = 6  # Trust averaging window
ϕ = 1.8  # Trust threshold sensitivity
P_crit = 6  # Number of predators to trigger zero-trust

# Clustering & routing
w1, w2, w3, w4, w5 = 0.4, 0.2, 0.2, 0.15, 0.05  # Fitness weights
w_h, w_q = 0.1, 0.05
α_e = 0.35
β_e = 6.0
ϵ = 0.12  # Firefly coupling
τ = 2.5   # Trust decay in firefly
r_chaos = 3.9  # Logistic map chaos parameter

# Sink position (center of body region)
P_s = np.array([2.5, 1.5])
d_max = 5.0

# Random seed
np.random.seed(42)
random.seed(42)

# ==============================
# DATA CLASSES
# ==============================

@dataclass
class QuantumState:
    alpha: complex
    beta: complex
    coherence: float

    def measure(self) -> str:
        prob_ch = abs(self.alpha)**2
        return "CH" if random.random() < prob_ch else "Member"

@dataclass
class Node:
    id: int
    pos: np.ndarray
    velocity: np.ndarray
    E: float
    H_rate: float
    T: float
    trust_history: List[float] = field(default_factory=list)
    DT: float = 1.0
    IT: float = 1.0
    S: float = 0.0  # Successful forwards
    F: float = 0.0  # Failed forwards
    PDR: float = 1.0
    role: str = "Member"
    cluster_id: Optional[int] = None
    isolated: bool = False
    phase: float = 0.0
    personal_best_fitness: float = 0.0
    personal_best_vector: np.ndarray = None
    quantum: QuantumState = None
    data_buffer: List[float] = field(default_factory=list)
    last_update: int = 0
    low_trust_streak: int = 0

    def __post_init__(self):
        d_to_sink = np.linalg.norm(self.pos - P_s)
        Q = np.exp(-γ * d_to_sink - δ * 0)
        alpha = np.sqrt((self.E / E_max) * ((d_max - d_to_sink) / d_max))
        beta = np.sqrt(1 - alpha**2)
        self.quantum = QuantumState(alpha, beta, Q)
        self.trust_history.append(self.T)
        self.personal_best_vector = np.array([self.E/E_max, d_to_sink/d_max, 0, self.T, 0])

# ==============================
# STEP 1: NETWORK INITIALIZATION
# ==============================

def initialize_network() -> List[Node]:
    nodes = []
    μ_x, μ_y = 2.5, 1.5
    σ_x, σ_y = 0.7, 0.6

    for i in range(NUM_NODES):
        x = np.random.normal(μ_x, σ_x)
        y = np.random.normal(μ_y, σ_y)
        pos = np.array([x, y])
        vel = np.array([np.random.normal(0, 0.05), np.random.normal(0, 0.05)])

        E = np.random.uniform(E_min, E_max)

        A_i = np.random.normal(A_mean, A_std)
        ΔT_i = np.random.normal(2.0, 0.5)
        motion_intensity = np.random.uniform(0.5, 1.5)
        P_bio = k_h_thermal * A_i * ΔT_i + k_h_piezo * A_i * motion_intensity
        H_rate = η_0 * P_bio

        T = 1.0

        node = Node(
            id=i,
            pos=pos,
            velocity=vel,
            E=E,
            H_rate=H_rate,
            T=T
        )

        # Beacon transmission cost
        d_to_sink = np.linalg.norm(pos - P_s)
        k_b = 32
        E_beacon = E_elec * k_b + ϵ_amp * k_b * d_to_sink**2
        node.E = max(0, node.E - E_beacon)

        nodes.append(node)
    return nodes

# ==============================
# STEP 2: QUANTUM-ENHANCED CLUSTERING
# ==============================

def fitness(node: Node, nodes: List[Node], t: int) -> float:
    E = node.E
    d_to_sink = np.linalg.norm(node.pos - P_s)
    R_t = R_0 * math.exp(-κ * t)
    neighbors = [n for n in nodes if np.linalg.norm(n.pos - node.pos) < R_t and not n.isolated]
    C = len(neighbors)
    ΔE = (node.E - E_min) if hasattr(node, '_prev_E') else 0
    if not hasattr(node, '_prev_E'):
        node._prev_E = node.E
    node._prev_E = node.E

    # Normalize
    f1 = E / E_max
    f2 = d_to_sink / d_max
    f3 = C / NUM_NODES
    f4 = node.T
    f5 = ΔE / E_max

    # Adaptive weights
    w1n = math.exp(0.1 * f1)
    w2n = math.exp(0.1 * (1 - f2))
    w3n = math.exp(0.1 * f3)
    w4n = math.exp(0.1 * f4)
    w5n = math.exp(0.1 * (1 - f5))
    total = w1n + w2n + w3n + w4n + w5n
    w1n /= total; w2n /= total; w3n /= total; w4n /= total; w5n /= total

    return w1n*f1 - w2n*f2 + w3n*f3 + w4n*f4 - w5n*f5

def update_qpsso(nodes: List[Node], t: int):
    CH_candidates = []
    mbest_vec = np.zeros(5)
    count = 0
    for node in nodes:
        if node.isolated:
            continue
        f = fitness(node, nodes, t)
        if f > node.personal_best_fitness:
            node.personal_best_fitness = f
            node.personal_best_vector = np.array([
                node.E/E_max,
                np.linalg.norm(node.pos - P_s)/d_max,
                len([n for n in nodes if np.linalg.norm(n.pos - node.pos) < R_0*math.exp(-κ*t) and not n.isolated])/NUM_NODES,
                node.T,
                (node.E - E_min)/E_max
            ])
        if node.personal_best_vector is not None:
            mbest_vec += node.personal_best_vector
            count += 1
    if count > 0:
        mbest_vec /= count

    beta_t = 0.6 * (1 - t / (NUM_ROUNDS + 10))
    E_avg = np.mean([n.E for n in nodes if not n.isolated]) + 1e-8
    theta = 0.7 * math.exp(-E_avg / E_max)

    for node in nodes:
        if node.isolated:
            continue
        u = random.random()
        d_to_sink = np.linalg.norm(node.pos - P_s)
        Q = math.exp(-γ * d_to_sink - δ * t)
        personal = node.personal_best_vector
        if personal is None:
            personal = np.random.rand(5)
        diff = np.abs(mbest_vec - personal)
        step = beta_t * diff * math.log(1/u) * Q
        new_vec = personal + (-1)**random.randint(1,2) * step

        f = fitness(node, nodes, t)
        max_f = max((fitness(n, nodes, t) for n in nodes if not n.isolated), default=1)
        prob_ch = f / (max_f + 1e-8)
        node.quantum.alpha = complex(math.sqrt(prob_ch))
        node.quantum.beta = complex(math.sqrt(1 - prob_ch))
        node.quantum.coherence = Q

        node.role = "CH" if prob_ch > theta else "Member"
        CH_candidates.append(node)

# ==============================
# STEP 3: TRUST ECOSYSTEM
# ==============================

def update_direct_trust(nodes: List[Node]):
    for node in nodes:
        if node.isolated:
            continue
        success_prob = 0.85 + 0.1 * (node.E / E_max) * node.T
        if random.random() < success_prob:
            node.S += ⋋_s * node.PDR
        else:
            node.F += ⋋_f * (1 - node.PDR)
        node.PDR = node.S / (node.S + node.F + 1e-6)
        node.DT = node.S / (node.S + node.F + 1e-6)

def update_indirect_trust(nodes: List[Node], t: int):
    for node in nodes:
        if node.isolated:
            continue
        neighbors = [n for n in nodes if np.linalg.norm(n.pos - node.pos) < R_0 and not n.isolated]
        if not neighbors:
            node.IT = node.DT
            continue
        total = 0.0
        weight_sum = 0.0
        for n in neighbors:
            d = np.linalg.norm(node.pos - n.pos)
            w_ij = math.exp(-d / R_0) * n.quantum.coherence
            xi = math.exp(-γ * d)
            it_contrib = n.DT * xi * w_ij
            total += it_contrib
            weight_sum += w_ij
        node.IT = total / (weight_sum + 1e-8)

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def neural_trust_update(nodes: List[Node]):
    for node in nodes:
        if node.isolated:
            continue
        x = np.array([
            node.DT,
            node.IT,
            node.E / E_max,
            node.PDR,
            node.H_rate / 100.0
        ])
        # Layer 1 (10 neurons)
        w1 = np.random.randn(10, 5) * 0.5
        b1 = np.random.randn(10) * 0.1
        h1 = np.array([sigmoid(aj) for aj in w1 @ x + b1])
        # Layer 2 (5 neurons)
        w2 = np.random.randn(5, 10) * 0.5
        b2 = np.random.randn(5) * 0.1
        h2 = np.array([sigmoid(aj) for aj in w2 @ h1 + b2])
        # Output
        v = np.random.randn(5) * 0.5
        T_pred = np.sum(v * h2)
        T_pred = max(0.1, min(1.0, T_pred))
        node.T = 0.8 * node.T + 0.2 * T_pred
        node.trust_history.append(node.T)

def predator_prey_trust_control(nodes: List[Node], t: int):
    active = [n for n in nodes if not n.isolated]
    if not active:
        return
    mu_T = np.mean([n.T for n in active])
    sigma_T = np.std([n.T for n in active])
    E_avg = np.mean([n.E for n in active])
    T_th = mu_T - ϕ * sigma_T * math.exp(-E_avg / E_max)

    for node in nodes:
        if node.isolated:
            continue
        if node.T < T_th:
            node.low_trust_streak += 1
            if node.low_trust_streak >= 5:
                node.isolated = True
                print(f"Node {node.id} isolated due to low trust at t={t}")
        else:
            node.low_trust_streak = 0

# ==============================
# STEP 4: INTRA-CLUSTER DATA FUSION
# ==============================

def fuse_cluster_data(cluster: List[Node]) -> Tuple[float, float]:
    if len(cluster) == 1:
        return np.random.normal(72, 1), 32  # HR reading

    readings = [np.random.normal(72, 1 + 0.5*(1-n.T)) for n in cluster]
    cov = np.cov(readings) if len(readings) > 1 else 0
    var = np.var(readings) + 1e-6
    rho_avg = 0
    for i, ni in enumerate(cluster):
        for j, nj in enumerate(cluster):
            if i != j:
                d = np.linalg.norm(ni.pos - nj.pos)
                rho = cov / var * math.exp(-d / 0.6)
                rho_avg += abs(rho)
    rho_avg = rho_avg / (len(cluster)*(len(cluster)-1)) if len(cluster) > 1 else 0

    weights = [n.E/E_max * n.T for n in cluster]
    weights = [w / sum(weights) for w in weights]
    fused = sum(w * r for w, r in zip(weights, readings))
    k_fused = 32 * (1 - rho_avg)
    E_fuse = E_proc * k_fused
    for n in cluster:
        n.E = max(0, n.E - E_fuse / len(cluster))
    return fused, k_fused

def quantum_compress(k: float) -> float:
    k_q = k / qubit_size * (1 + 0.1)  # Overhead
    return k_q

# ==============================
# STEP 5: SWARM ROUTING
# ==============================

def firefly_sync(nodes: List[Node], t: int):
    CHs = [n for n in nodes if n.role == "CH" and not n.isolated]
    if len(CHs) < 2:
        return
    for ch in CHs:
        coupling = 0.0
        for ch2 in CHs:
            if ch.id == ch2.id:
                continue
            d = np.linalg.norm(ch.pos - ch2.pos)
            trust_diff = abs(ch.T - ch2.T)
            intensity = (ch.E * ch.T) / (d + 0.1) * math.exp(-trust_diff / τ)
            coupling += intensity * math.sin(ch2.phase - ch.phase)
        noise = np.random.normal(0, 0.05)
        d_phase = 2 * math.pi * 0.5 + ϵ * coupling + noise
        ch.phase += d_phase * TIME_STEP
        ch.phase %= (2 * math.pi)

def fractal_routing_path(nodes: List[Node], src: Node) -> List[Node]:
    CHs = [n for n in nodes if n.role == "CH" and not n.isolated]
    if not CHs:
        return []
    x = (hash(src.id + int(np.sum(src.pos))) % 1000) / 1000
    B = int(r_chaos * len(CHs) * x * (1 - x))
    B = max(1, min(B, 5))
    path = [src]
    current = src
    for _ in range(B):
        candidates = [n for n in CHs if n not in path]
        if not candidates:
            break
        next_hop = min(candidates, key=lambda n: np.linalg.norm(n.pos - P_s))
        path.append(next_hop)
    return path

# ==============================
# STEP 6: BIO-ENERGY HARVESTING
# ==============================

def update_harvesting(nodes: List[Node], t: int):
    for node in nodes:
        if node.isolated:
            continue
        A_i = np.random.normal(A_mean, A_std)
        ΔT_i = 2.0 + 0.5 * math.sin(2 * math.pi * t / 10) + np.random.normal(0, 0.2)
        motion = 1.0 + 0.8 * math.sin(2 * math.pi * t / 8)
        P_bio = k_h_thermal * A_i * ΔT_i + k_h_piezo * A_i * motion
        η = η_0 * (1 - math.exp(-node.E / E_th))
        noise = np.random.normal(0, 0.08)
        H_rate = η * P_bio + noise
        H_rate = max(0, H_rate)
        energy_harvested = H_rate * TIME_STEP
        charge_eff = (1 - node.E / E_cap)
        node.E = min(E_cap, node.E + energy_harvested * charge_eff)
        node.E = max(0, node.E - E_leak_rate)
        node.H_rate = H_rate

# ==============================
# STEP 7: SELF-EVOLUTION
# ==============================

def energy_entropy(nodes: List[Node]) -> float:
    energies = [n.E for n in nodes if not n.isolated]
    if sum(energies) == 0:
        return 0
    p = np.array(energies) / sum(energies)
    p = p[p > 0]
    return -np.sum(p * np.log2(p + 1e-8))

def should_recluster(nodes: List[Node], t: int) -> bool:
    if not hasattr(should_recluster, 'prev_entropy'):
        should_recluster.prev_entropy = energy_entropy(nodes)
        return False
    curr_entropy = energy_entropy(nodes)
    delta = curr_entropy - should_recluster.prev_entropy
    should_recluster.prev_entropy = curr_entropy
    sigma_E = np.std([n.E for n in nodes])
    E_th = (α_e * E_max / NUM_NODES) * (1 - math.exp(-sigma_E / E_max))
    return delta > E_th

def activate_zero_trust(nodes: List[Node]):
    predators = [n for n in nodes if n.T < 0.5]
    if len(predators) > P_crit:
        CHs = [n for n in nodes if n.role == "CH" and not n.isolated]
        for ch in CHs:
            k = 32
            E_enc = k * E_elec * math.log2(qubit_size) * (1 + 0.15)
            E_err = E_qproc * error_correction_factor * k
            ch.E = max(0, ch.E - E_enc - E_err)

# ==============================
# MAIN SIMULATION LOOP
# ==============================

def simulate_quanbiotrust():
    nodes = initialize_network()
    history = {
        'energy': [], 'trust': [], 'chs': [], 'isolated': [], 'entropy': []
    }

    for t in range(NUM_ROUNDS):
        print(f"\n[Round {t}] Node Count: {len(nodes)}, Active: {len([n for n in nodes if not n.isolated])}")

        # Step 6: Harvesting
        update_harvesting(nodes, t)

        # Step 2: Clustering
        update_qpsso(nodes, t)

        # Step 3: Trust
        update_direct_trust(nodes)
        update_indirect_trust(nodes, t)
        neural_trust_update(nodes)
        predator_prey_trust_control(nodes, t)

        # Cluster formation
        CHs = [n for n in nodes if n.role == "CH" and not n.isolated]
        for node in nodes:
            if node in CHs:
                node.cluster_id = node.id
            else:
                if CHs:
                    closest = min(CHs, key=lambda ch: np.linalg.norm(node.pos - ch.pos))
                    node.cluster_id = closest.id

        # Step 4: Data fusion
        for ch in CHs:
            cluster = [n for n in nodes if n.cluster_id == ch.id and not n.isolated]
            if cluster:
                fused, k_raw = fuse_cluster_data(cluster)
                k_quantum = quantum_compress(k_raw)
                ch.data_buffer.append(fused)

        # Step 5: Routing
        firefly_sync(nodes, t)
        for ch in CHs:
            path = fractal_routing_path(nodes, ch)
            hop_cost = E_elec * 32 + ϵ_amp * 32 * 2.0**2
            ch.E = max(0, ch.E - hop_cost * len(path))

        # Step 7: Self-evolution
        if should_recluster(nodes, t):
            print(f"[Re-clustering] at t={t}")
            for ch in CHs:
                ch.E = max(0, ch.E - E_proc * 100)

        activate_zero_trust(nodes)

        # Record stats
        active = [n for n in nodes if not n.isolated]
        history['energy'].append(np.mean([n.E for n in active]) if active else 0)
        history['trust'].append(np.mean([n.T for n in active]) if active else 0)
        history['chs'].append(len(CHs))
        history['isolated'].append(len([n for n in nodes if n.isolated]))
        history['entropy'].append(energy_entropy(nodes))

    return nodes, history

# ==============================
# VISUALIZATION
# ==============================

def plot_results(history):
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 2, 1)
    plt.plot(history['energy'], label='Avg Energy')
    plt.title("Energy Evolution")
    plt.xlabel("Round")
    plt.ylabel("Energy (J)")
    plt.grid(True)

    plt.subplot(3, 2, 2)
    plt.plot(history['trust'], color='green', label='Trust')
    plt.axhline(0.5, color='red', linestyle='--', label='Trust Threshold')
    plt.title("Trust Evolution")
    plt.xlabel("Round")
    plt.ylabel("Trust Score")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(history['chs'], color='orange')
    plt.title("Active Cluster Heads")
    plt.xlabel("Round")
    plt.ylabel("Count")
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(history['isolated'], color='red')
    plt.title("Isolated Nodes")
    plt.xlabel("Round")
    plt.ylabel("Count")
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(history['entropy'], color='purple')
    plt.title("Energy Entropy")
    plt.xlabel("Round")
    plt.ylabel("Entropy")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    print("🚀 Starting QuanBioTrust Advanced Simulation...")
    final_nodes, history = simulate_quanbiotrust()
    print("✅ Simulation Complete.")
    plot_results(history)