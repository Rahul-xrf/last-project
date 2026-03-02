"""
Generate a comprehensive PDF walkthrough report for project submission.
MARL Dynamic Spectrum Allocation with SUMO Integration.
"""

import os
import json
from fpdf import FPDF

RESULTS_DIR = "results"


class SubmissionPDF(FPDF):
    """Professional PDF for project submission."""

    def header(self):
        if self.page_no() == 1:
            return  # No header on title page
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "MARL Dynamic Spectrum Allocation | Project Report", align="C")
        self.ln(4)
        self.set_draw_color(41, 128, 185)
        self.set_line_width(0.4)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, number, title):
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(41, 128, 185)
        self.ln(4)
        self.cell(0, 12, f"{number}. {title}")
        self.ln(14)
        self.set_draw_color(41, 128, 185)
        self.set_line_width(0.8)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(6)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(52, 73, 94)
        self.ln(3)
        self.cell(0, 8, title)
        self.ln(10)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(44, 62, 80)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text, indent=10):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(44, 62, 80)
        x = self.get_x()
        self.set_x(x + indent)
        self.cell(5, 5.5, "-")
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def bold_bullet(self, label, text, indent=10):
        x = self.get_x()
        self.set_x(x + indent)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(44, 62, 80)
        self.cell(5, 5.5, "-")
        self.set_font("Helvetica", "B", 10)
        self.cell(self.get_string_width(label) + 2, 5.5, label)
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 // len(headers)] * len(headers)
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(41, 128, 185)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(44, 62, 80)
        fill = False
        for row in rows:
            if fill:
                self.set_fill_color(235, 245, 251)
            else:
                self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6.5, str(cell), border=1, fill=True, align="C")
            self.ln()
            fill = not fill
        self.ln(4)

    def code_block(self, text):
        self.set_font("Courier", "", 9)
        self.set_fill_color(245, 245, 245)
        self.set_text_color(44, 62, 80)
        self.set_draw_color(200, 200, 200)
        x = self.get_x()
        y = self.get_y()
        lines = text.strip().split('\n')
        block_h = len(lines) * 5.5 + 4
        self.rect(x, y, 190, block_h)
        self.set_xy(x + 3, y + 2)
        for line in lines:
            self.cell(0, 5.5, line)
            self.ln(5.5)
            self.set_x(x + 3)
        self.ln(4)


def generate_submission_pdf():
    pdf = SubmissionPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # =========================================================================
    # TITLE PAGE
    # =========================================================================
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font("Helvetica", "B", 30)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(0, 14, "Multi-Agent Reinforcement", align="C")
    pdf.ln(14)
    pdf.cell(0, 14, "Learning for Dynamic", align="C")
    pdf.ln(14)
    pdf.cell(0, 14, "Spectrum Allocation", align="C")
    pdf.ln(18)

    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(127, 140, 141)
    pdf.cell(0, 10, "in Cognitive Internet of Vehicles (CIoV)", align="C")
    pdf.ln(25)

    pdf.set_draw_color(41, 128, 185)
    pdf.set_line_width(1.5)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(20)

    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, "Project Report", align="C")
    pdf.ln(15)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, "Technologies: SUMO | PyTorch | DRQN | MARL | CTDE | Dec-POMDP", align="C")
    pdf.ln(8)
    pdf.cell(0, 7, "Framework: Python 3 | Deep Recurrent Q-Network", align="C")
    pdf.ln(30)

    pdf.set_draw_color(200, 200, 200)
    pdf.set_line_width(0.3)
    pdf.line(30, pdf.get_y(), 180, pdf.get_y())

    # =========================================================================
    # TABLE OF CONTENTS
    # =========================================================================
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(41, 128, 185)
    pdf.cell(0, 12, "Table of Contents")
    pdf.ln(15)

    toc_items = [
        ("1", "Introduction & Problem Statement"),
        ("2", "System Architecture"),
        ("3", "SUMO Traffic Simulation"),
        ("4", "DRQN Agent Design"),
        ("5", "Training Methodology (CTDE)"),
        ("6", "Evaluation & Baselines"),
        ("7", "Results & Analysis"),
        ("8", "Implementation Guide"),
        ("9", "Conclusion & Future Work"),
    ]
    for num, title in toc_items:
        pdf.set_font("Helvetica", "", 12)
        pdf.set_text_color(44, 62, 80)
        pdf.cell(10, 8, num + ".")
        pdf.cell(0, 8, title)
        pdf.ln(8)

    # =========================================================================
    # 1. INTRODUCTION
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("1", "Introduction & Problem Statement")

    pdf.body_text(
        "The explosive growth of connected vehicles in smart cities demands efficient "
        "wireless communication. In a Cognitive Internet of Vehicles (CIoV) environment, "
        "multiple electric vehicles must share limited radio spectrum resources for "
        "safety-critical communications including V2V (Vehicle-to-Vehicle) and V2I "
        "(Vehicle-to-Infrastructure) messaging."
    )

    pdf.body_text(
        "This project addresses the Dynamic Spectrum Allocation (DSA) problem using "
        "Multi-Agent Reinforcement Learning (MARL). Instead of relying on static "
        "pre-planned channel assignments, each vehicle agent learns through experience "
        "to dynamically select optimal wireless channels, adapting to changing traffic "
        "conditions and minimizing packet collisions."
    )

    pdf.section_title("Problem Formulation")
    pdf.body_text(
        "The DSA problem is formulated as a Decentralized Partially Observable Markov "
        "Decision Process (Dec-POMDP) where:"
    )
    pdf.bold_bullet("Agents: ", "10 electric vehicles on an urban road network")
    pdf.bold_bullet("Actions: ", "Select one of 5 available wireless channels")
    pdf.bold_bullet("Observations: ", "Local channel occupancy, interference levels, "
                    "vehicle state (position, speed), previous action, agent ID")
    pdf.bold_bullet("Reward: ", "+1 for successful sole-channel transmission, "
                    "-1 for collision, +0.5 for successful shared transmission")
    pdf.bold_bullet("Objective: ", "Maximize successful packet delivery while "
                    "minimizing channel collisions through learned coordination")

    pdf.section_title("Key Evaluation Metrics")
    metrics_table = [
        ["Metric", "Description", "Target"],
        ["Throughput", "Total successful transmissions", "Maximize"],
        ["Packet Delivery Ratio", "Successful / Total transmissions", "Maximize"],
        ["Average Latency", "Mean delivery time per packet", "Minimize"],
        ["Collision Rate", "Collisions / Total transmissions", "Minimize"],
    ]
    pdf.add_table(metrics_table[0], metrics_table[1:], col_widths=[50, 95, 45])

    # =========================================================================
    # 2. SYSTEM ARCHITECTURE
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("2", "System Architecture")

    pdf.body_text(
        "The system follows a modular architecture with clear separation of concerns. "
        "Each module handles a specific aspect of the MARL pipeline."
    )

    pdf.section_title("Module Overview")
    modules = [
        ["Module", "File(s)", "Responsibility"],
        ["Configuration", "config.py", "Centralized hyperparameters"],
        ["Simulation Env", "environment.py", "CIoV simulation + SUMO integration"],
        ["Agent Design", "drqn_model.py, agent.py", "DRQN network + RL agent"],
        ["Training", "train.py", "CTDE training loop"],
        ["Evaluation", "evaluate.py", "Baseline comparison"],
        ["Visualization", "visualize.py", "Plot generation"],
        ["Entry Point", "main.py", "Pipeline orchestration"],
        ["PDF Report", "generate_pdf.py", "Report generation"],
        ["SUMO Config", "sumo_config/", "Traffic network & routes"],
    ]
    pdf.add_table(modules[0], modules[1:], col_widths=[40, 65, 85])

    pdf.section_title("Data Flow Pipeline")
    pdf.body_text("The system operates in three sequential phases:")
    pdf.ln(2)
    pdf.bold_bullet("Phase 1 - Training: ", "SUMO provides realistic vehicle dynamics. "
                    "Agents observe channel states, select actions via DRQN, receive "
                    "rewards, and update weights using experience replay.")
    pdf.bold_bullet("Phase 2 - Evaluation: ", "Trained MARL agent is benchmarked "
                    "against Static Allocation and Random Allocation baselines "
                    "over 100 evaluation episodes.")
    pdf.bold_bullet("Phase 3 - Visualization: ", "Publication-quality plots are "
                    "generated comparing all methods across 4 key metrics.")

    pdf.section_title("Technology Stack")
    tech_table = [
        ["Technology", "Version", "Purpose"],
        ["Python", "3.x", "Core programming language"],
        ["PyTorch", "2.x", "Deep learning framework"],
        ["SUMO", "1.26.0", "Traffic simulation (TraCI API)"],
        ["NumPy", "1.x", "Numerical computation"],
        ["Matplotlib", "3.x", "Visualization and plotting"],
        ["tqdm", "4.x", "Progress bar for training"],
    ]
    pdf.add_table(tech_table[0], tech_table[1:], col_widths=[50, 30, 110])

    # =========================================================================
    # 3. SUMO INTEGRATION
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("3", "SUMO Traffic Simulation")

    pdf.body_text(
        "SUMO (Simulation of Urban Mobility) is an open-source, microscopic traffic "
        "simulator developed by DLR. It provides realistic vehicle movement including "
        "acceleration, deceleration, lane changing, and traffic light compliance. "
        "The project integrates SUMO via the TraCI (Traffic Control Interface) API."
    )

    pdf.section_title("Network Configuration")
    pdf.body_text("The urban environment is defined by three SUMO configuration files:")
    pdf.ln(2)

    sumo_files = [
        ["File", "Description"],
        ["urban.net.xml", "4x4 urban grid network with traffic lights, 2-lane roads, 200m spacing"],
        ["urban.rou.xml", "10 vehicle routes with validated connectivity, repeating paths"],
        ["urban.sumocfg", "Master config linking network + routes, 1.0s step length"],
    ]
    pdf.add_table(sumo_files[0], sumo_files[1:], col_widths=[50, 140])

    pdf.section_title("TraCI Integration")
    pdf.body_text(
        "The SUMOSpectrumEnvironment class interfaces with SUMO through TraCI to "
        "retrieve real-time vehicle data at each simulation step:"
    )
    pdf.bullet("traci.vehicle.getPosition() - Real 2D coordinates from SUMO")
    pdf.bullet("traci.vehicle.getSpeed() - Actual vehicle speed (m/s)")
    pdf.bullet("traci.simulationStep() - Advance SUMO by one time step")
    pdf.bullet("traci.start() / traci.close() - Lifecycle management per episode")

    pdf.body_text(
        "Each training episode starts a fresh SUMO simulation. The environment waits "
        "until all 10 vehicles are deployed, then runs 200 time steps of concurrent "
        "spectrum allocation decisions."
    )

    pdf.section_title("Dual-Mode Operation")
    pdf.body_text(
        "The project supports two operation modes controlled by config.py:"
    )
    pdf.bold_bullet("SUMO Mode (USE_SUMO=True): ", "Realistic vehicle dynamics from "
                    "SUMO. Positions, speeds, and movements are governed by the traffic "
                    "simulator. Distance-based interference model uses actual coordinates.")
    pdf.bold_bullet("Standalone Mode (USE_SUMO=False): ", "Simple grid-based movement "
                    "for fast prototyping. No SUMO installation required.")

    # =========================================================================
    # 4. DRQN AGENT DESIGN
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("4", "DRQN Agent Design")

    pdf.body_text(
        "The intelligent agent uses a Deep Recurrent Q-Network (DRQN), which extends "
        "DQN by incorporating LSTM layers for temporal reasoning. This enables agents "
        "to make decisions based on a history of observations, not just the current state."
    )

    pdf.section_title("Network Architecture")
    arch_table = [
        ["Layer", "Type", "Details"],
        ["1. Agent ID Embedding", "nn.Embedding", "10 agents -> 32-dim vector"],
        ["2. Input Processing", "nn.Linear + ReLU", "(obs_dim + 32) -> 64 hidden"],
        ["3. Temporal Memory", "nn.LSTM", "64 -> 64, 1 layer"],
        ["4. Output", "nn.Linear", "64 -> 5 (Q-values per channel)"],
    ]
    pdf.add_table(arch_table[0], arch_table[1:], col_widths=[55, 45, 90])

    pdf.section_title("Observation Space (30 dimensions)")
    obs_table = [
        ["Component", "Dimensions", "Description"],
        ["Channel Occupancy", "5", "Normalized user count per channel"],
        ["Interference Levels", "5", "Signal interference per channel"],
        ["Position (x, y)", "2", "Normalized vehicle coordinates"],
        ["Speed", "1", "Normalized current speed"],
        ["Remaining Payload", "1", "Packets left to transmit"],
        ["Time Budget", "1", "Steps remaining for delivery"],
        ["Previous Action", "5", "One-hot encoding of last channel"],
        ["Agent ID", "10", "One-hot encoding of vehicle identity"],
    ]
    pdf.add_table(obs_table[0], obs_table[1:], col_widths=[50, 30, 110])

    pdf.section_title("Key Design Features")
    pdf.bold_bullet("Parameter Sharing: ", "All 10 agents share one DRQN network, "
                    "reducing memory and enabling faster training.")
    pdf.bold_bullet("Agent ID Embedding: ", "A learned 32-dimensional embedding per "
                    "agent ID allows differentiated channel preferences despite shared weights.")
    pdf.bold_bullet("Per-Agent Hidden States: ", "Each agent maintains its own LSTM "
                    "hidden state, enabling independent temporal reasoning.")
    pdf.bold_bullet("Epsilon-Greedy Exploration: ", "Starts at 1.0, decays by 0.995 "
                    "per episode to minimum 0.05, balancing exploration vs exploitation.")

    # =========================================================================
    # 5. TRAINING
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("5", "Training Methodology (CTDE)")

    pdf.body_text(
        "The training follows the Centralized Training with Distributed Execution (CTDE) "
        "paradigm. During training, all agent experiences are collected in a shared "
        "replay buffer and used to update a single DRQN. During execution (evaluation), "
        "each agent acts independently using only local observations."
    )

    pdf.section_title("Training Hyperparameters")
    hp_table = [
        ["Parameter", "Value", "Description"],
        ["Episodes", "1000", "Total training episodes"],
        ["Steps per Episode", "200", "Max time steps per episode"],
        ["Learning Rate", "0.001", "Adam optimizer LR"],
        ["Discount Factor", "0.95", "Future reward discount"],
        ["Batch Size", "32", "Replay sampling batch size"],
        ["Buffer Size", "5000", "Experience replay capacity"],
        ["Sequence Length", "8", "LSTM sequence length"],
        ["Target Update (soft)", "0.01", "Polyak averaging coefficient"],
        ["Target Update (hard)", "Every 10 eps", "Full weight copy interval"],
        ["Epsilon Decay", "0.995", "Per-episode exploration decay"],
        ["Hidden Size", "64", "DRQN hidden layer width"],
        ["Warmup Episodes", "64", "Explore before updating"],
    ]
    pdf.add_table(hp_table[0], hp_table[1:], col_widths=[50, 35, 105])

    pdf.section_title("Training Algorithm")
    pdf.body_text("For each episode:")
    pdf.bullet("1. Reset SUMO simulation and agent hidden states")
    pdf.bullet("2. For each time step (200 steps):")
    pdf.bullet("   a. Each agent observes local channel state", indent=15)
    pdf.bullet("   b. DRQN selects channel (epsilon-greedy)", indent=15)
    pdf.bullet("   c. Environment computes rewards based on collisions", indent=15)
    pdf.bullet("   d. Store (obs, action, reward, next_obs, done) in episode memory", indent=15)
    pdf.bullet("3. Add episode sequences to shared replay buffer")
    pdf.bullet("4. Sample random batches and update DRQN via MSE loss")
    pdf.bullet("5. Soft-update target network (Polyak averaging)")
    pdf.bullet("6. Decay exploration rate and save checkpoints")

    # =========================================================================
    # 6. EVALUATION
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("6", "Evaluation & Baselines")

    pdf.body_text(
        "The trained MARL agent is compared against two baseline spectrum allocation "
        "methods over 100 evaluation episodes to measure its effectiveness."
    )

    pdf.section_title("Baseline Methods")
    pdf.bold_bullet("Static Allocation: ", "Each vehicle is permanently assigned a "
                    "channel based on its ID (vehicle_id % num_channels). This mimics "
                    "traditional pre-planned spectrum allocation with no adaptation.")
    pdf.ln(2)
    pdf.bold_bullet("Random Allocation: ", "Each vehicle randomly selects a channel "
                    "at every time step with uniform probability. This serves as a "
                    "lower-bound benchmark with zero intelligence.")
    pdf.ln(2)
    pdf.bold_bullet("MARL (DRQN): ", "Our trained agent using the best checkpoint "
                    "model. Agent acts greedily (epsilon=0) using learned Q-values "
                    "conditioned on local observations and agent identity.")

    pdf.section_title("Evaluation Methodology")
    pdf.body_text(
        "Each method is evaluated under identical conditions:"
    )
    pdf.bullet("100 evaluation episodes per method")
    pdf.bullet("Same SUMO network and vehicle routes")
    pdf.bullet("200 time steps per episode")
    pdf.bullet("Metrics averaged across all episodes with standard deviation")
    pdf.bullet("MARL agent uses best performing checkpoint (greedy policy)")

    # =========================================================================
    # 7. RESULTS
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("7", "Results & Analysis")

    pdf.section_title("Training Progress (1000 Episodes)")
    pdf.body_text(
        "The training curves below show the MARL agent's learning progress over "
        "1000 episodes with SUMO-integrated environment."
    )

    training_img = os.path.join(RESULTS_DIR, "training_curves.png")
    if os.path.exists(training_img):
        pdf.image(training_img, x=15, w=180)
        pdf.ln(3)

    pdf.body_text(
        "Key observations: The reward shows learning dynamics with initial exploration, "
        "a dip during the exploration phase, and recovery as the policy stabilizes. "
        "PDR remains high at ~1.0 throughout training. The collision rate stays near "
        "zero, indicating successful channel coordination. Epsilon decays smoothly "
        "from 1.0 to 0.05."
    )

    # Performance comparison page
    pdf.add_page()
    pdf.section_title("Performance Comparison: MARL vs Baselines")

    # Load actual results
    eval_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            results = json.load(f)

        marl = results.get("MARL (DRQN)", {})
        static = results.get("Static", {})
        random_r = results.get("Random", {})

        def fmt(d, key, dec=4):
            return f'{d.get(key, {}).get("mean", 0):.{dec}f}'

        comparison = [
            ["Metric", "MARL (DRQN)", "Static", "Random"],
            ["Throughput", fmt(marl, "throughput", 1),
             fmt(static, "throughput", 1), fmt(random_r, "throughput", 1)],
            ["Packet Delivery Ratio", fmt(marl, "packet_delivery_ratio"),
             fmt(static, "packet_delivery_ratio"), fmt(random_r, "packet_delivery_ratio")],
            ["Average Latency", fmt(marl, "avg_latency"),
             fmt(static, "avg_latency"), fmt(random_r, "avg_latency")],
            ["Collision Rate", fmt(marl, "collision_rate"),
             fmt(static, "collision_rate"), fmt(random_r, "collision_rate")],
        ]
        pdf.add_table(comparison[0], comparison[1:], col_widths=[50, 45, 45, 45])

    metric_img = os.path.join(RESULTS_DIR, "metric_comparison.png")
    if os.path.exists(metric_img):
        pdf.image(metric_img, x=15, w=180)
        pdf.ln(3)

    pdf.body_text(
        "The comparison shows all three methods achieve excellent performance in the "
        "SUMO environment. The MARL agent successfully learns to coordinate channel "
        "selection, matching the performance of baseline methods while maintaining "
        "the crucial advantage of dynamic adaptability to changing conditions."
    )

    # Episode trends page
    pdf.add_page()
    pdf.section_title("Per-Episode Performance Trends")

    trends_img = os.path.join(RESULTS_DIR, "episode_trends.png")
    if os.path.exists(trends_img):
        pdf.image(trends_img, x=15, w=180)
        pdf.ln(3)

    pdf.body_text(
        "The per-episode trends demonstrate consistent performance across all 100 "
        "evaluation episodes for all three methods. MARL shows stable throughput "
        "and PDR with minimal variance, confirming the reliability of the learned "
        "policy. The latency variation is natural and reflects different traffic "
        "conditions across SUMO simulation runs."
    )

    # =========================================================================
    # 8. IMPLEMENTATION GUIDE
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("8", "Implementation Guide")

    pdf.section_title("Prerequisites")
    pdf.bullet("Python 3.8 or higher")
    pdf.bullet("SUMO 1.26.0 (with SUMO_HOME environment variable set)")
    pdf.bullet("PyTorch, NumPy, Matplotlib, tqdm, fpdf2")

    pdf.section_title("Installation")
    pdf.code_block("pip install torch numpy matplotlib tqdm fpdf2")

    pdf.section_title("Running the Full Pipeline")
    pdf.code_block("python main.py --train --evaluate --visualize --episodes 1000")
    pdf.body_text(
        "This single command trains the MARL agent (1000 episodes), evaluates against "
        "baselines (100 episodes each), and generates comparison plots."
    )

    pdf.section_title("Individual Commands")
    cmds = [
        ["Command", "Description"],
        ["python main.py --train --episodes 1000", "Train MARL agent only"],
        ["python main.py --evaluate --eval-episodes 100", "Evaluate against baselines"],
        ["python main.py --visualize", "Generate plots from saved data"],
        ["python generate_pdf.py", "Generate this PDF report"],
        ["sumo-gui -c sumo_config/urban.sumocfg", "Open SUMO GUI to view traffic"],
    ]
    pdf.add_table(cmds[0], cmds[1:], col_widths=[95, 95])

    pdf.section_title("Configuration Options (config.py)")
    cfg_table = [
        ["Setting", "Default", "Description"],
        ["USE_SUMO", "True", "Enable SUMO integration"],
        ["SUMO_GUI", "False", "Show SUMO GUI window"],
        ["NUM_VEHICLES", "10", "Number of vehicle agents"],
        ["NUM_CHANNELS", "5", "Available wireless channels"],
        ["NUM_EPISODES", "500", "Default training episodes"],
        ["LEARNING_RATE", "0.001", "DRQN learning rate"],
    ]
    pdf.add_table(cfg_table[0], cfg_table[1:], col_widths=[45, 30, 115])

    pdf.section_title("Output Files")
    out_table = [
        ["File", "Description"],
        ["results/training_curves.png", "Training progress (reward, PDR, collision, epsilon)"],
        ["results/metric_comparison.png", "Bar chart: MARL vs Static vs Random"],
        ["results/episode_trends.png", "Per-episode performance trends"],
        ["results/training_history.json", "Raw training metrics (all episodes)"],
        ["results/evaluation_results.json", "Raw evaluation data (mean, std, per-episode)"],
        ["models/best_model.pth", "Best performing model checkpoint"],
        ["models/final_model.pth", "Final model after training completes"],
        ["results/walkthrough_report.pdf", "This PDF report"],
    ]
    pdf.add_table(out_table[0], out_table[1:], col_widths=[70, 120])

    # =========================================================================
    # 9. CONCLUSION
    # =========================================================================
    pdf.add_page()
    pdf.chapter_title("9", "Conclusion & Future Work")

    pdf.section_title("Summary")
    pdf.body_text(
        "This project successfully demonstrates the application of Multi-Agent "
        "Reinforcement Learning to the Dynamic Spectrum Allocation problem in a "
        "Cognitive Internet of Vehicles environment. The key accomplishments are:"
    )
    pdf.ln(2)
    pdf.bullet("Implemented a complete MARL system using DRQN with LSTM for "
               "temporal reasoning in spectrum allocation decisions")
    pdf.bullet("Integrated SUMO traffic simulator providing realistic urban "
               "vehicle dynamics via the TraCI API")
    pdf.bullet("Agent ID embedding enables parameter-shared agents to learn "
               "differentiated per-vehicle channel preferences")
    pdf.bullet("CTDE architecture allows centralized training with shared "
               "experience while maintaining distributed execution")
    pdf.bullet("Comprehensive evaluation pipeline compares MARL against "
               "Static and Random baselines across 4 key metrics")
    pdf.bullet("Automated visualization generates publication-quality plots "
               "for all performance metrics")
    pdf.bullet("Modular, configurable codebase with dual-mode operation "
               "(SUMO-integrated and standalone)")

    pdf.section_title("Future Work")
    pdf.bullet("Implement QMIX or VDN value decomposition for better "
               "joint action learning in dense scenarios")
    pdf.bullet("Add inter-agent communication channels for explicit "
               "coordination during spectrum allocation")
    pdf.bullet("Scale to larger networks (50+ vehicles, 20+ channels) "
               "and test scalability limits")
    pdf.bullet("Incorporate dynamic traffic scenarios with varying "
               "vehicle density and emergency situations")
    pdf.bullet("Add more sophisticated channel models including "
               "fading, shadowing, and Doppler effects")
    pdf.bullet("Integrate with real-world V2X communication standards "
               "(C-V2X, DSRC/802.11p)")
    pdf.bullet("Deploy on edge computing infrastructure for "
               "real-time spectrum allocation decisions")

    # Save PDF
    output_path = os.path.join(RESULTS_DIR, "walkthrough_report.pdf")
    pdf.output(output_path)
    print(f"\nPDF generated successfully: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"Total pages: {pdf.page_no()}")
    return output_path


if __name__ == "__main__":
    generate_submission_pdf()
