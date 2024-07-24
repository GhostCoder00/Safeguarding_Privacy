for seed in 0 1 2 3 4
do
    project=colorado
    lr=-4
    python main_non_FL.py -p $project -c_lr $lr -seed $seed

    project=korea
    lr=-2
    python main_non_FL.py -p $project -c_lr $lr -seed $seed

    project=engagenet
    lr=-4
    python main_non_FL.py -p $project -c_lr $lr -seed $seed

    project=daisee
    lr=-4
    python main_non_FL.py -p $project -c_lr $lr -seed $seed
done