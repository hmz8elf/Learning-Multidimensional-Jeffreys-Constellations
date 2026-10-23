# Archive

One-off scripts kept for traceability. They are not needed to reproduce the results and may not run as-is (paths changed when the repository was restructured).

- `approach2/update_*.py`, `append.py`, `scratch_extract*.py`: edited or generated parts of the progress report.
- `approach2/patch_score.py`, `patch_threads.py`, `root/patch_pam.py`, `root/patch_report.py`: patches already applied to the main code.
- `approach2/pilot_*.py`, `profile_svgd_5000.py`, `dry_run.py`: timing pilots before the full runs (`pilot_score.py` passes wrong arguments to the λ* solver).
- `approach2/evaluate_svgd_metrics.py`: old SVGD protocol (N = 1000, 3 seeds), replaced by `evaluate_svgd_metrics_gpu.py`.
- `approach1/validate_step5.py`: older variant of `approach1/step5.py`.
- `approach2/PROGRESS_REPORT.md`: previous version of the report (the current one is `approach2/docs/PROGRESS_REPORT.md`).
- `*/plan*.txt`, `*/livrables.txt`: internship planning notes (French).
- `root/generer_figures_nb.py`: black-and-white figures for the written report (writes outside this repository).
