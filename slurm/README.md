# SLURM Quickstart (Cluster)

These scripts are intended for cluster execution with data in `$SCRATCH/prueba_slabs_data`.

## 0) One-time setup: manifest file (no sbatch env vars required)

```bash
cd ~/prueba_slabs
cp slurm/manifest.env.example slurm/manifest.env
nano slurm/manifest.env
```

Set at least:
- `PROJECT_DIR`
- `DATA_DIR`
- `OPENGATE_PYTHON`

## 1) Generate MC data with array job (1000 samples, 200 concurrent)

```bash
cd ~/prueba_slabs
mkdir -p logs
sbatch slurm/gen_mc_multinoise_array.slurm
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

## Optional local execution (no SLURM)

```bash
cd ~/prueba_slabs
source .venv/bin/activate
export PYTHONPATH=$PWD/src

python scripts/extend_mc_multinoise_from_existing.py \
	--mc-root "$SCRATCH/prueba_slabs_data/mc_runs_opengate_photon_1000" \
	--pairs-root "$SCRATCH/prueba_slabs_data/mc_runs_opengate_photon_1000_multinoise_pairs" \
	--dataset-out "$SCRATCH/prueba_slabs_data/data_real_photon_1000_multinoise" \
	--max-parallel 8 \
	--clean-pairs

python scripts/train.py \
	--data-root "$SCRATCH/prueba_slabs_data/data_real_photon_1000_multinoise" \
	--out-dir "$SCRATCH/prueba_slabs_data/artifacts_real_photon_1000_multinoise_e20_mi210" \
	--epochs 20 \
	--batch-size 2 \
	--device cuda \
	--workers 8 \
	--amp \
	--loss-alpha 3.0 \
	--patience 5 \
	--min-delta 0.0 \
	--save-every 5 \
	--input-norm-mode per_channel_max \
	--input-dose-scale 1.0 \
	--output-activation softplus
```
