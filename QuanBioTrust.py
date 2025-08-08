# quanbiotrust.py
# Quantum-Enhanced, Bio-Energy-Harvesting Trust Framework for WBAN
# Author: QuanBioTrust Simulation Engine
# Date: 2025

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.neural_network import MLPRegressor
import random
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# ==============================
# CONFIGURATION & CONSTANTS
# ==============================

# Physical and network parameters
NUM_NODES = 30
E_min = 0.5    # Joules
E_max = 2.0
E_cap = 2.5    # Supercapacitor limit
E_th = 0.1     # Harvester saturation threshold
η_0 = 0.8      # Max harvesting efficiency
k_h = 0.01     # Thermal constant (µW/cm²/K)
R_0 = 3.0      # Communication range (m)
d_max = 5.0    # Max distance in network
T_cycle = 10   # Bio-rhythm cycle (e.g., breath)
κ = 0.05       # Range decay
γ = 0.01       # Decoherence coefficient
δ = 0.001      # Time decay in coherence
δ_d = 0.5      # Spatial correlation decay
δ_t = 2.0      # Temporal correlation decay
T_window = 5   # Trust time window
Δt = 1.0       # Time slot duration (seconds)

# Energy constants (µJ/bit)
E_elec = 50e-6
ϵ_amp = 100e-6
E_proc = 20e-6
E_qproc = 150e-6
ϵ_m = 10e-6    # Mobility energy

# Weights
w1, w2, w3, w4, w5 = 0.4, 0.2, 0.2, 0.15, 0.05  # Fitness weights
w_h, w_q = 0.1, 0.05                             # Harvesting & quantum weights
η = 1.0                                            # Trust softmax parameter
α_e = 0.3                                          # Energy entropy sensitivity
β_e = 5.0                                          # Re-clustering sensitivity

# Firefly & routing
ϵ = 0.1       # Coupling strength
τ = 2.0       # Trust decay constant
P_crit = 5    # Predator threshold for zero-trust
p_crit = 0.59 # Percolation threshold

# Zero-trust
qubit_size = 5  # 32-bit key ~ 5 qubits (2^5 = 32)
D_dec = 0.1     # Decoherence default

# Sink position
P_s = np.array([2.5, 1.5])

# Random seed
np.random.seed(42)
random.seed(42)

# ==============================
# DATA CLASSES
# ==============================

@dataclass
class Node:
    id: int
    pos: np.ndarray
    E: float
    H_rate: float
    T: float
    Q: float
    alpha: float
    beta: float
    role: str  # 'CH' or 'Member'
    cluster_id: Optional[int] = None
    DT: float = 1.0
    IT: float = 1.0
    S: float = 0.0
    F: float = 0.0
    PDR: float = 1.0
    velocity: np.ndarray = None
    personal_best_fitness: float = 0.0
    personal_best_pos: np.ndarray = None
    phase: float = 0.0  # For firefly
    isolated: bool = False

# ==============================
# STEP 1: NETWORK INITIALIZATION
# ==============================

def initialize_network() -> List[Node]:
    nodes = []
    for i in range(NUM_NODES):
        # Position (body surface: e.g., chest-centered)
        x = np.random.normal(2.5, 0.8)
        y = np.random.normal(1.5, 0.6)
        pos = np.array([x, y])
        d_to_sink = np.linalg.norm(pos - P_s)

        # Initial energy
        E = np.random.uniform(E_min, E_max)

        # Harvesting rate
        A_i = np.random.uniform(1.0, 2.0)  # cm²
        ΔT_i = np.random.normal(2.0, 0.5)
        P_bio = k_h * A_i * ΔT_i
        H_rate = η_0 * P_bio

        # Quantum coherence
        Q = np.exp(-γ * d_to_sink)

        # Quantum superposition
        alpha = np.sqrt((E / E_max) * ((d_max - d_to_sink) / d_max))
        beta = np.sqrt(1 - alpha**2)

        # Beacon energy
        k_b = 32  # bits
        E_beacon = E_elec * k_b + ϵ_amp * k_b * d_to_sink**2
        E -= E_beacon
        E = max(0, min(E, E_cap))

        # Entanglement with sink (simulated, not stored)

        nodes.append(Node(
            id=i,
            pos=pos,
            E=E,
            H_rate=H_rate,
            T=1.0,
            Q=Q,
            alpha=alpha,
            beta=beta,
            role="Member",
            velocity=np.array([0.0, 0.0]),
            phase=2 * np.pi * random.random()
        ))
    return nodes

# ==============================
# STEP 2: QUANTUM-ENHANCED CLUSTERING
# ==============================

def fitness(node: Node, nodes: List[Node], t: int) -> float:
    E = node.E
    d_to_sink = np.linalg.norm(node.pos - P_s)
    R_t = R_0 * np.exp(-κ * t)
    neighbors = [n for n in nodes if np.linalg.norm(n.pos - node.pos) < R_t and n.id != node.id]
    C = len(neighbors)
    C_max = NUM_NODES
    ΔE = node.E - E_min  # Simplified depletion
    T = node.T

    f1 = E / E_max
    f2 = d_to_sink / d_max
    f3 = C / C_max
    f4 = T
    f5 = ΔE / E_max

    w1_adj = np.exp(0.1 * f1)
    w2_adj = np.exp(0.1 * (1 - f2))
    w3_adj = np.exp(0.1 * f3)
    w4_adj = np.exp(0.1 * f4)
    w5_adj = np.exp(0.1 * (1 - f5))
    total = w1_adj + w2_adj + w3_adj + w4_adj + w5_adj
    w1n = w1_adj / total
    w2n = w2_adj / total
    w3n = w3_adj / total
    w4n = w4n_adj / total
    w5n = w5_adj / total

    return w1n*f1 - w2n*f2 + w3n*f3 + w4n*f4 - w5n*f5

def update_qpsso(nodes: List[Node], t: int):
    personal_bests = []
    for node in nodes:
        f = fitness(node, nodes, t)
        if f > node.personal_best_fitness:
            node.personal_best_fitness = f
            node.personal_best_pos = np.array([
                node.E / E_max,
                np.linalg.norm(node.pos - P_s) / d_max,
                len([n for n in nodes if np.linalg.norm(n.pos - node.pos) < R_0 * np.exp(-κ*t)]) / NUM_NODES,
                node.T,
                (node.E - E_min) / E_max
            ])
        personal_bests.append(node.personal_best_pos if node.personal_best_pos is not None
                              else np.random.rand(5))

    if not personal_bests:
        return

    mbest = np.mean(personal_bests, axis=0)
    E_avg = np.mean([n.E for n in nodes])
    theta = 0.7 * np.exp(-E_avg / E_max)

    for node in nodes:
        beta_t = 0.5 + 0.5 * (1 - t / 100)  # Decreasing
        u = random.random()
        Q = np.exp(-γ * np.linalg.norm(node.pos - P_s) - δ * t)
        new_pos = node.personal_best_pos + (-1)**np.random.randint(1,3) * beta_t * \
                  np.abs(mbest - np.array([node.E/E_max, np.linalg.norm(node.pos-P_s)/d_max, 
                                           len([n for n in nodes if np.linalg.norm(n.pos-node.pos)<R_0])/NUM_NODES,
                                           node.T, (node.E-E_min)/E_max])) * math.log(1/u) * Q
        # Not updating physical pos, just internal state
        f = fitness(node, nodes, t)
        prob_ch = f / (1e-8 + max(fitness(n, nodes, t) for n in nodes))
        node.alpha = np.sqrt(prob_ch)
        node.beta = np.sqrt(1 - prob_ch)
        node.role = "CH" if prob_ch > theta else "Member"

# ==============================
# STEP 3: TRUST ECOSYSTEM
# ==============================

def update_direct_trust(nodes: List[Node]):
    for node in nodes:
        # Simulate packet success/failure
        if random.random() < 0.9:  # 90% success
            node.S += 1.0
        else:
            node.F += 1.0
        PDR = node.S / (node.S + node.F + 1e-6)
        node.DT = node.S / (node.S + node.F + 1e-6)
        node.PDR = PDR

def update_indirect_trust(nodes: List[Node]):
    for node in nodes:
        neighbors = [n for n in nodes if np.linalg.norm(n.pos - node.pos) < R_0 and n.id != node.id]
        if not neighbors:
            node.IT = node.DT
            continue
        total = 0.0
        weight_sum = 0.0
        for n in neighbors:
            d = np.linalg.norm(node.pos - n.pos)
            w_ij = np.exp(-d / R_0) * n.Q
            xi = np.exp(-γ * d)
            it_contrib = n.DT * xi * w_ij
            total += it_contrib
            weight_sum += w_ij
        node.IT = total / (weight_sum + 1e-8) if weight_sum > 0 else node.DT

def deep_trust_reasoning(nodes: List[Node]):
    model = MLPRegressor(hidden_layer_sizes=(10, 5), max_iter=200, alpha=1e-4)
    X_train = []
    y_train = []
    for node in nodes:
        X_train.append([
            node.DT,
            node.IT,
            node.E / E_max,
            node.PDR,
            node.H_rate / 100.0
        ])
        y_train.append(node.T)
    if len(X_train) > 5:
        model.fit(X_train, y_train)
        predictions = model.predict(X_train)
        for i, node in enumerate(nodes):
            node.T = 0.7 * node.T + 0.3 * predictions[i]  # Smooth update

def predator_prey_trust_control(nodes: List[Node], t: int):
    mu_T = np.mean([n.T for n in nodes])
    sigma_T = np.std([n.T for n in nodes])
    E_avg = np.mean([n.E for n in nodes])
    T_th = mu_T - 1.5 * sigma_T * np.exp(-E_avg / E_max)

    for node in nodes:
        if node.T < T_th:
            if not hasattr(node, '_low_trust_start'):
                node._low_trust_start = t
            elif t - node._low_trust_start > 5:  # 5 seconds
                node.isolated = True
        else:
            if hasattr(node, '_low_trust_start'):
                del node._low_trust_start

# ==============================
# STEP 4: INTRA-CLUSTER DATA COLLECTION
# ==============================

def fuse_data_in_cluster(cluster_nodes: List[Node]) -> float:
    readings = [np.random.normal(72, 2) for _ in cluster_nodes]  # Simulated HR
    correlations = []
    for i, ni in enumerate(cluster_nodes):
        for j, nj in enumerate(cluster_nodes):
            if i != j:
                d = np.linalg.norm(ni.pos - nj.pos)
                rho = np.exp(-d / δ_d) * np.exp(-abs(0) / δ_t)  # No time diff
                correlations.append(rho)
    avg_corr = np.mean(correlations) if correlations else 0
    fused = np.mean(readings) * (1 - avg_corr)
    k_fused = 32 * (1 - avg_corr)
    E_fusion = E_proc * k_fused
    for node in cluster_nodes:
        node.E = max(0, node.E - E_fusion / len(cluster_nodes))
    return fused

def quantum_compress(data: float) -> float:
    # Simulate compression: 32 bits -> 5 qubits
    k_quantum = 32 / qubit_size * (1 + 0.1)  # Overhead
    return k_quantum

# ==============================
# STEP 5: SWARM-DRIVEN MULTI-HOP ROUTING
# ==============================

def synchronize_fireflies(nodes: List[Node], t: int):
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
            intensity = (ch.E * ch.T) / (d + 0.1) * np.exp(-trust_diff / τ)
            coupling += intensity * np.sin(ch2.phase - ch.phase)
        noise = np.random.normal(0, 0.05)
        d_phase = 2*np.pi*0.5 + ϵ * coupling + noise
        ch.phase += d_phase * Δt
        ch.phase = ch.phase % (2*np.pi)

def select_next_hop(current: Node, CHs: List[Node]) -> Optional[Node]:
    scores = []
    for ch in CHs:
        if ch.id == current.id:
            continue
        d = np.linalg.norm(ch.pos - P_s)
        latency = d / 2.0
        w_e = ch.E / E_max
        w_t = ch.T
        w_d = 1 - d / d_max
        w_l = 1 - latency / 10.0
        score = 0.4*w_e + 0.3*w_t - 0.2*w_d - 0.1*w_l
        scores.append((score, ch))
    if not scores:
        return None
    _, best = max(scores, key=lambda x: x[0])
    return best

# ==============================
# STEP 6: BIO-ENERGY HARVESTING
# ==============================

def update_harvesting(nodes: List[Node], t: int):
    for node in nodes:
        A_i = np.random.uniform(1.0, 2.0)
        ΔT_i = 2.0 + 0.5 * np.sin(2 * np.pi * t / T_cycle)
        G_i = np.random.uniform(0.8, 1.2)
        P_bio = k_h * A_i * ΔT_i * G_i
        η = η_0 * (1 - np.exp(-node.E / E_th))
        noise = np.random.normal(0, 0.05)
        H_rate = η * P_bio + noise
        node.H_rate = max(0, H_rate)
        harvested = node.H_rate * Δt
        node.E = min(E_cap, node.E + harvested * (1 - node.E / E_cap))
        node.E = max(0, node.E - 1e-6)  # Leakage

# ==============================
# STEP 7: SELF-EVOLUTION & MAINTENANCE
# ==============================

def should_recluster(nodes: List[Node], t: int) -> bool:
    energies = [n.E for n in nodes]
    p = np.array(energies) / (sum(energies) + 1e-8)
    p = p[p > 0]
    H_E = -np.sum(p * np.log2(p))
    if not hasattr(should_recluster, 'prev_H'):
        should_recluster.prev_H = H_E
        return False
    delta_H = H_E - should_recluster.prev_H
    sigma_E = np.std(energies)
    E_th = (α_e * E_max / NUM_NODES) * (1 - np.exp(-sigma_E / E_max))
    should_recluster.prev_H = H_E
    return delta_H > E_th

def activate_zero_trust(nodes: List[Node], CHs: List[Node]):
    predators = [n for n in nodes if n.T < 0.5]
    if len(predators) > P_crit:
        print(f"[Security] Zero-trust mode activated: {len(predators)} malicious nodes.")
        for ch in CHs:
            k = 32  # bits
            E_enc = k * E_elec * np.log2(qubit_size) * (1 + D_dec)
            E_err = E_qproc * 3 * k
            ch.E = max(0, ch.E - E_enc - E_err)

# ==============================
# MAIN SIMULATION LOOP
# ==============================

def simulate_quanbiotrust(num_rounds: int = 100):
    nodes = initialize_network()
    history = {'energy': [], 'trust': [], 'live_chs': []}

    for t in range(num_rounds):
        print(f"\n--- Round {t} ---")

        # Step 1: Already initialized
        # Step 6: Harvesting
        update_harvesting(nodes, t)

        # Step 2: Clustering
        update_qpsso(nodes, t)

        # Step 3: Trust
        update_direct_trust(nodes)
        update_indirect_trust(nodes)
        deep_trust_reasoning(nodes)
        predator_prey_trust_control(nodes, t)

        # Form clusters
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
            cluster = [n for n in nodes if n.cluster_id == ch.id]
            if len(cluster) > 1:
                fused = fuse_data_in_cluster(cluster)
                k_quantum = quantum_compress(fused)

        # Step 5: Routing
        synchronize_fireflies(nodes, t)
        for ch in CHs:
            next_hop = select_next_hop(ch, CHs)
            if next_hop:
                d = np.linalg.norm(ch.pos - next_hop.pos)
                α = 2 if d < 2 else 4
                E_hop = E_elec * 32 + ϵ_amp * 32 * d**α
                ch.E = max(0, ch.E - E_hop)

        # Step 7: Self-evolution
        if should_recluster(nodes, t):
            print("[Re-clustering] Triggered due to energy imbalance.")
            for ch in CHs:
                ch.E = max(0, ch.E - E_proc * 100)

        activate_zero_trust(nodes, CHs)

        # Record metrics
        avg_energy = np.mean([n.E for n in nodes])
        avg_trust = np.mean([n.T for n in nodes if not n.isolated])
        live_chs = len([n for n in CHs if not n.isolated])
        history['energy'].append(avg_energy)
        history['trust'].append(avg_trust)
        history['live_chs'].append(live_chs)

    return history

# ==============================
# RUN SIMULATION & PLOT
# ==============================

if __name__ == "__main__":
    print("Starting QuanBioTrust Simulation...")
    history = simulate_quanbiotrust(num_rounds=100)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history['energy'])
    plt.title("Avg Energy Over Time")
    plt.xlabel("Round")
    plt.ylabel("Energy (J)")

    plt.subplot(1, 3, 2)
    plt.plot(history['trust'])
    plt.title("Avg Trust Over Time")
    plt.xlabel("Round")
    plt.ylabel("Trust Score")

    plt.subplot(1, 3, 3)
    plt.plot(history['live_chs'])
    plt.title("Active CHs Over Time")
    plt.xlabel("Round")
    plt.ylabel("Number of CHs")

    plt.tight_layout()
    plt.show()

    print("Simulation completed.")