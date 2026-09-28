# Dependency and resource notes

- `requirements-cpu.txt`: exact CPU-side versions recorded by the final source package.
- `requirements-gpu.txt`: transformer stack bounds; install PyTorch separately for the server CUDA version.
- Python: 3.10+; the recorded fine-tuning runtime used PyTorch 2.6.0+cu124.
- External model IDs: `intfloat/multilingual-e5-small`, `intfloat/multilingual-e5-base`, and optional `Qwen/Qwen2.5-1.5B`.
- Full production run: Linux, 32–64 CPU cores and 128 GB RAM preferred; modern 24 GB+ NVIDIA GPU for neural stages; 100 GB free scratch space recommended.
- CPU-only baseline: 16+ cores, 64 GB RAM, about 25 GB scratch space.

Review model/dependency licenses and competition data rules before public distribution.

