# Analysis environment handoff

The isolated `.venv-analysis` is installed and verified: Python 3.12.13,
NumPy 2.3.3, h5py 3.14.0, Pydantic 2.12.3, SciPy 1.16.3, NiBabel 5.3.2 and
PyTorch 2.10.0+cpu. The receipt is `artifacts/analysis-environment.json`.
The annotation environment was not modified.

The integrated code and verification boundary are documented in
[analysis.md](analysis.md). Spatial acquisition and mapping have completed for
all nine participants. Numerical/interface checks use actual released textual
features and compiled queries. Scientific brain/model fits remain unexecuted.

To recheck dependency imports without downloading, in Windows CMD:

```cmd
cd /d C:\Users\pigby\neurosymbolic_coognitive_linguistics
call scripts\setup_analysis.cmd --check
```

To inspect actual input availability without fitting:

```cmd
.venv-analysis\Scripts\python.exe -B scripts\run_analysis.py preflight
```

Current blockers are the not-yet-compiled `semantics-v5` snapshot and full measured/
model arrays being on the cluster. There is no need to reinstall packages.
After annotation completion, `scripts\prepare_analysis.cmd` compiles, verifies and
packages the final snapshot; `scripts\upload_analysis.cmd` transfers files without
executing remote commands.

Cluster setup remains user-operated. `setup_analysis.sbatch` builds a separate
CUDA-capable environment only on an allocated compute node under project storage.
Existing extraction and annotation environments remain separate. Installation
and scientific fitting are long user-run jobs.
