import json

with open('mi_curve_results.json', 'r') as f:
    data = json.load(f)

print("M=256 (max MI):")
for nt in ["nt2", "nt4", "nt8"]:
    if "256" in data[nt]["flow"]:
        flow_mean = data[nt]["flow"]["256"]["mean"]
    else:
        flow_mean = "N/A"
    
    if "svgd" in data[nt] and "256" in data[nt]["svgd"]:
        svgd_mean = data[nt]["svgd"]["256"]["mean"]
    else:
        svgd_mean = "N/A"
    print(f"{nt}: Flow={flow_mean}, SVGD={svgd_mean}")
