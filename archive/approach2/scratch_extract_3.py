import json

with open('mi_curve_results.json', 'r') as f:
    mi_data = json.load(f)

with open('evaluation_metrics.json', 'r') as f:
    flow_metrics = json.load(f)

with open('svgd_metrics.json', 'r') as f:
    svgd_metrics = json.load(f)

with open('score_metrics.json', 'r') as f:
    score_metrics = json.load(f)

# Combine into table
table = "| $n_t$ ($d$) | Méthode | MMD mean ± std | Energy Dist mean ± std | ESS/N | MI (M=256) |\n"
table += "|---|---|---|---|---|---|\n"

nts = [2, 4, 8]

for nt in nts:
    d = nt * 2
    
    # FLOW
    flow_m = next(m for m in flow_metrics if m["n_t"] == nt)
    mi_flow = mi_data[f"nt{nt}"]["flow"]["256"]["mean"]
    table += f"| **$n_t={nt}$ ($d={d}$)**  | **Flow**  | ${flow_m['MMD_mean']:.6f} \\pm {flow_m['MMD_std']:.6f}$ | ${flow_m['Energy_Distance_mean']:.6f} \\pm {flow_m['Energy_Distance_std']:.6f}$ | ${flow_m['ESS_N_mean']:.4f}$ | ${mi_flow:.2f}$ bits |\n"
    
    # SVGD
    svgd_m = next(m for m in svgd_metrics if m["n_t"] == nt)
    mi_svgd = mi_data[f"nt{nt}"]["svgd"]["256"]["mean"]
    table += f"|                       | **SVGD**  | ${svgd_m['MMD_mean']:.6f} \\pm {svgd_m['MMD_std']:.6f}$ | ${svgd_m['Energy_Distance_mean']:.6f} \\pm {svgd_m['Energy_Distance_std']:.6f}$ | N/A      | ${mi_svgd:.2f}$ bits |\n"
    
    # SCORE
    score_m = next(m for m in score_metrics if m["n_t"] == nt)
    mi_score = mi_data[f"nt{nt}"]["score"]["256"]["mean"]
    table += f"|                       | **Score** | ${score_m['MMD_mean']:.6f} \\pm {score_m['MMD_std']:.6f}$ | ${score_m['Energy_Distance_mean']:.6f} \\pm {score_m['Energy_Distance_std']:.6f}$ | N/A      | ${mi_score:.2f}$ bits |\n"

print(table)
