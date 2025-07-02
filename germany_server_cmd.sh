# slurm template
srun --gres=gpu:a100:1 --account=researchers -p a100_only --pty bash
sbatch --ntasks=1 --cpus-per-task=4 --account=researchers -p a100_only --wrap='echo CPU Test Started; sleep 60; echo CPU Test Completed'

# my code
srun --gres=gpu:a100:1 --account=researchers -p a100_only --job-name="MW+nFL" --cpus-per-task=4 --time=2-00:00:00 bash germany_nFL_lr_search.sh
srun --gres=gpu:a100:1 --account=researchers -p a100_only --job-name="MW+FL" --cpus-per-task=4 --time=2-00:00:00 bash germany_FL_lr_search.sh
srun --gres=gpu:a100:1 --account=researchers -p a100_only --job-name="MW+bag" --cpus-per-task=4 --time=2-00:00:00 bash bagging_lr_search.sh