# Physics-Informed 3D Dose Denoising (OpenGATE, Slab Phantom)

This repository implements an end-to-end Monte Carlo (MC) + deep learning pipeline for denoising low-statistics beamlet dose volumes with a 3D ResUNet.

## Scope

- Input: low-statistics dose + SPR map (2 channels)
- Target: high-fidelity dose (100k events), normalized by target max
- Phantom: multilayer slab (water, cortical bone, lung, water)
- Validation: central axis Bragg peak, lateral penumbra, and 3D gamma pass rate

## Project structure

- `scripts/generate_synthetic_dataset.py`: creates synthetic training pairs in `.npz`
- `scripts/run_mc_campaign.py`: runs base real MC campaign (low/high)
- `scripts/extend_mc_multinoise_from_existing.py`: extends existing 2k/100k campaigns to multinoise (2k/5k/10k/20k -> 100k)
- `scripts/run_multinoise_campaign.py`: generates multinoise pairs from scratch
- `scripts/build_dataset_from_mc.py`: converts real MC outputs to training `.npz`
- `scripts/train.py`: trains the 3D ResUNet with physics-weighted loss
- `scripts/validate.py`: computes physical validation metrics on test split
- `src/proton_denoise/physics.py`: slab phantom + synthetic dose/noise model
- `src/proton_denoise/model.py`: ResUNet3D
- `src/proton_denoise/losses.py`: exponentially weighted MSE
- `src/proton_denoise/metrics.py`: Bragg, penumbra, gamma metrics
- `mc/campaign.example.json`: MC campaign config template
- `mc/templates/run_proton_sim.sh.example`: wrapper template for your GATE/Geant4 app

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
```

## Run

1. Option A: generate synthetic dataset (quick baseline):

```bash
python scripts/generate_synthetic_dataset.py --root data --seed 42
```

2. Option B: generate real Monte Carlo dataset (base 2k/100k):

2.1 Copy templates and adapt command wrapper to your simulator binary.

```bash
cp mc/templates/run_proton_sim.sh.example mc/run_proton_sim.sh
chmod +x mc/run_proton_sim.sh
cp mc/campaign.example.json mc/campaign.json
```

Edit `mc/campaign.json` and set:
- `simulator_command_template`, usually:
	- `./mc/run_proton_sim.sh --energy {energy_mev} --events {events} --out {output_dir} --seed {seed}`

2.2 Run MC campaign (creates `mc_runs/<sample>/low` and `mc_runs/<sample>/high`):

```bash
python scripts/run_mc_campaign.py --config mc/campaign.json
```

2.3 Build training dataset from MC outputs:

```bash
python scripts/build_dataset_from_mc.py --mc-root mc_runs --out-root data
```

3. Multinoise extension (recommended):

Extend existing base MC samples from `2k -> 100k` to `2k,5k,10k,20k -> 100k` without re-running expensive 100k targets:

```bash
python scripts/extend_mc_multinoise_from_existing.py \
	--mc-root mc_runs_opengate_photon_1000 \
	--pairs-root mc_runs_opengate_photon_1000_multinoise_pairs \
	--dataset-out data_real_photon_1000_multinoise \
	--max-parallel 8 \
	--clean-pairs
```

This builds:
- `mc_runs_opengate_photon_1000_multinoise_pairs` (one pair per sample+noise-level)
- `data_real_photon_1000_multinoise` (train/val/test `.npz`)

4. Train:

3. Train:

```bash
python scripts/train.py --data-root data_real_photon_1000_multinoise --out-dir artifacts_real_photon_1000_multinoise_e20 --epochs 20 --batch-size 2 --device cuda --workers 4 --amp --loss-alpha 3.0 --save-every 5 --input-norm-mode per_channel_max --output-activation softplus
```

5. Validate physical metrics:

```bash
python scripts/validate.py --data-root data_real_photon_1000_multinoise --checkpoint artifacts_real_photon_1000_multinoise_e20/best_model.pt --out-dir artifacts_real_photon_1000_multinoise_e20/validation_latest --device cuda --input-norm-mode per_channel_max
```

## Cluster scaling to 10k multinoise pairs

Target: `10,000` training pairs with noise levels `{2k,5k,10k,20k}` and reference `100k`.

Formula:
- `num_pairs = num_base_samples * num_noise_levels`
- `10,000 = num_base_samples * 4` => `num_base_samples = 2,500`

Practical plan:
1. Generate `2,500` base MC samples (`2k/100k`) on cluster workers.
2. Run `extend_mc_multinoise_from_existing.py` to create missing `5k/10k/20k` lows.
3. Build final multinoise dataset.

Suggested parallel/chunk strategy:
- Split base sample IDs by ranges (example: `0-249`, `250-499`, ...).
- Run each chunk on a node with its own output folder.
- Merge chunk folders into a single `mc_root` before extension step.

## Publish to GitHub

After cloning this folder to your machine, run:

```bash
git init
git add .
git commit -m "Initial commit: MC denoising pipeline with multinoise extension"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

## Notes

- For real MC, each run must export `dose.npy`; `spr.npy` is also expected in this workflow.
- Gamma implementation is brute-force and can be slow for large volumes.
