import re
with open("Approche 2/mi_curve.py", "r") as f:
    content = f.read()

# Replace get_pam_cartesian
old_pam_func = """def get_pam_cartesian(M: int, d: int, P_total: float) -> np.ndarray:
    \"\"\"
    Constellation PAM Cartésienne de taille ~M en dimension d.
    M_1d = round(M^(1/d)), taille effective = M_1d^d.
    Puissance uniformément répartie : P_total/d par dimension.
    \"\"\"
    M_1d    = max(2, round(M ** (1.0 / d)))
    P_pd    = P_total / d
    pam_1d  = get_pam_1d(M_1d, P_pd)
    grids   = [pam_1d] * d
    points  = np.array(list(product(*grids)))   # (M_1d^d, d)
    return points"""

new_pam_func = """def get_pam_cartesian(M: int, d: int, P_total: float, rng: np.random.Generator = None) -> np.ndarray:
    \"\"\"
    Constellation PAM Cartésienne de taille ~M en dimension d.
    M_1d = max(2, round(M**(1/d))).
    
    LIMITATION DOCUMENTÉE (d>=16) : Pour M <= 256, M_1d vaut toujours 2. 
    La grille sous-jacente est une PAM binaire par axe (65536 points).
    Pour garder une taille M, on sous-échantillonne M points au hasard parmi cette grille.
    Ainsi, la densité augmente avec M, mais la résolution de la grille reste fixe.
    \"\"\"
    if rng is None:
        rng = np.random.default_rng(42)
        
    M_1d    = max(2, round(M ** (1.0 / d)))
    P_pd    = P_total / d
    pam_1d  = get_pam_1d(M_1d, P_pd)
    
    # Si la grille complète est immense, on évite le memory error
    # Mais ici max M=256, donc max d=16, M_1d=2 -> 65536 points, ça passe en RAM.
    grids   = [pam_1d] * d
    points  = np.array(list(product(*grids)))   # (M_1d^d, d)
    
    # Sous-échantillonnage si la grille est plus grande que M
    if points.shape[0] > M:
        idx = rng.choice(points.shape[0], size=M, replace=False)
        points = points[idx]
        
    return points"""

content = content.replace(old_pam_func, new_pam_func)

old_pam_loop = """            # ── PAM Cartésien ───────────────────────────────────────
            if str(M) not in all_results[key].get("pam", {}):
                X_pam  = get_pam_cartesian(M, d, P)
                M_eff  = X_pam.shape[0]
                rng_pam = np.random.default_rng(42 + 100 * M + 10000 * nt)
                t0 = time.time()
                mi_pam = compute_mi_ba(X_pam, SIGMA2, N_Y, rng_pam)
                print(f"    PAM   M_eff={M_eff}  MI={mi_pam:.4f} bits  ({time.time()-t0:.1f}s)")
                all_results[key]["pam"][str(M)] = {
                    "mean": mi_pam, "std": 0.0, "M_effective": M_eff,
                }
                save_results(all_results)
            else:
                mi_pam = all_results[key]["pam"][str(M)]["mean"]
                M_eff  = all_results[key]["pam"][str(M)]["M_effective"]
                print(f"    PAM   M_eff={M_eff}  MI={mi_pam:.4f} bits [cached]")"""

new_pam_loop = """            # ── PAM Cartésien (avec répétitions si sous-échantillonnage) ──
            mi_pam_seeds = []
            pam_dict = all_results[key].get("pam", {})
            already_done_pam = pam_dict.get(str(M), {}).get("seeds", [])
            
            for seed in range(N_SEEDS):
                if seed < len(already_done_pam):
                    mi = already_done_pam[seed]
                    print(f"    PAM   seed={seed}: MI={mi:.4f} bits [cached]")
                    mi_pam_seeds.append(mi)
                    continue

                rng_pam = np.random.default_rng(42 + seed + 100 * M + 10000 * nt)
                X_pam = get_pam_cartesian(M, d, P, rng=rng_pam)
                M_eff = X_pam.shape[0]
                
                t0 = time.time()
                mi_pam = compute_mi_ba(X_pam, SIGMA2, N_Y, rng_pam)
                mi_pam_seeds.append(mi_pam)
                print(f"    PAM   seed={seed} M_eff={M_eff}: MI={mi_pam:.4f} bits ({time.time()-t0:.1f}s)")
                
                if str(M) not in all_results[key]["pam"]:
                    all_results[key]["pam"][str(M)] = {}
                all_results[key]["pam"][str(M)]["seeds"] = mi_pam_seeds[:]
                save_results(all_results)

            mi_pam_mean = float(np.mean(mi_pam_seeds))
            mi_pam_std  = float(np.std(mi_pam_seeds))
            all_results[key]["pam"][str(M)] = {
                "mean": mi_pam_mean, "std": mi_pam_std, "seeds": mi_pam_seeds, "M_effective": M_eff
            }
            print(f"    PAM → {mi_pam_mean:.4f} ± {mi_pam_std:.4f} bits")"""

content = content.replace(old_pam_loop, new_pam_loop)

# Fix missing initialisation if it exists
if '"pam": {}' in content and 'all_results[key] = {"M_values": M_VALUES, "flow": {}, "pam": {}}' not in content:
    pass

with open("Approche 2/mi_curve.py", "w") as f:
    f.write(content)

print("Patched mi_curve.py")
