srun --time=00:05:00 \
  --nodes=1 \
  --partition=gpu \
  --cpus-per-task=1 \
  --ntasks=1 \
  --gpus-per-task=1 \
  --export=ALL \
  bash -l -c "module load compiler/NVHPC && ./cudaScan -m scan -n 8"
