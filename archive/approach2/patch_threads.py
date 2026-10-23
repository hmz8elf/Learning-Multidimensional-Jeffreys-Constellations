import os

for file in ["evaluate_svgd_metrics.py", "run_mi_curve_svgd.py"]:
    with open(file, "r") as f:
        content = f.read()
    
    if "torch.set_num_threads" not in content:
        content = content.replace("import torch", "import torch\ntorch.set_num_threads(2)\n")
    
    # Also change n_samples to 1000 in metrics
    if file == "evaluate_svgd_metrics.py":
        content = content.replace("n_samples: int = 2000", "n_samples: int = 1000")
        
    with open(file, "w") as f:
        f.write(content)
