# Constitutional AI Project

Looking to implement and learn more about CAI with a focus on education

## Goal

1. Fine-tune an open source model to behave more like a TA
2. Quantify improvement against some benchmark (possibly MT Bench)


## Usage

To run a job on the HPC cluster, use the provided job scripts in the `jobs/` directory. You can submit jobs using `sbatch` and monitor them with `squeue`.

For example, to generate data, you can use the `generate_data.sh` script:

```bash
sbatch jobs/generate_data.sh
```

> [!NOTE] Make sure to submit the job from the root directory of the project to ensure correct paths.

