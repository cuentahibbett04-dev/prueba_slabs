# SLURM Quickstart (Cluster)

These scripts are intended for cluster execution with data in `$SCRATCH/prueba_slabs_data`.

## 1) Generate MC data with array job (1000 samples, 200 concurrent)

```bash
cd ~/prueba_slabs
mkdir -p logs
sbatch --export=ALL,OPENGATE_PYTHON=/path/to/opengate_env/bin/python slurm/gen_mc_multinoise_array.slurm
```

Progress checks:

```bash
MC_ROOT=$SCRATCH/prueba_slabs_data/mc_runs_opengate_photon_1000
echo "high: $(find "$MC_ROOT" -path '*/high/dose.npy' | wc -l)/1000"
echo "2k  : $(find "$MC_ROOT" -path '*/low/dose.npy' | wc -l)/1000"
echo "5k  : $(find "$MC_ROOT" -path '*/low_e05000/dose.npy' | wc -l)/1000"
echo "10k : $(find "$MC_ROOT" -path '*/low_e10000/dose.npy' | wc -l)/1000"
echo "20k : $(find "$MC_ROOT" -path '*/low_e20000/dose.npy' | wc -l)/1000"
```

## 2) Build multinoise pairs + dataset

```bash
cd ~/prueba_slabs
sbatch slurm/build_multinoise_dataset.slurm
```

Expected dataset output: `$SCRATCH/prueba_slabs_data/data_real_photon_1000_multinoise`

## 3) Train on MI210 (ROCm)

```bash
cd ~/prueba_slabs
sbatch slurm/train_mi210_rocm.slurm
```

Expected training output: `$SCRATCH/prueba_slabs_data/artifacts_real_photon_1000_multinoise_e20_mi210`
