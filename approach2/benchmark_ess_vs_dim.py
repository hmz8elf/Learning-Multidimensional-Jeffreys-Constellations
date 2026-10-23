import matplotlib.pyplot as plt

def main():
    # Recorded values from validation scripts
    # 1D Non-Gaussian ESS
    ess_n_1d = 0.9228
    
    # MIMO ESS for nt=2, 4, 8 (dims 4, 8, 16)
    dims = [4, 8, 16]
    nt_vals = [2, 4, 8]
    ess_n_mimo = [0.9943, 0.9846, 0.9885]
    
    plt.figure(figsize=(8, 5))
    plt.plot(dims, ess_n_mimo, marker='o', linestyle='-', color='b', label='MIMO Isotropic (RealNVP)')
    
    # Also plot the 1D non-gaussian for reference as a horizontal line or a point
    plt.axhline(y=ess_n_1d, color='r', linestyle='--', label='1D Non-Gaussian (Reference)')
    
    plt.title('ESS/N vs Dimension for MIMO (RealNVP Flow)')
    plt.xlabel('Dimension (d = 2 * n_t)')
    plt.ylabel('ESS / N')
    plt.xticks(dims)
    plt.ylim(0.0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.savefig('benchmark_ess_vs_dim.png')
    print("Saved benchmark_ess_vs_dim.png")

if __name__ == "__main__":
    main()
