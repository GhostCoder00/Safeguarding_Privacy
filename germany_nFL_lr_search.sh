printf "\nRunning Germany non-FL learning rate search\n"

project=germany

for lr in -5 -4.5 -4 -3.5 -3 -2.5 -2
do
    for seed in 0 1 2 3 4
    do
        python main_non_FL.py -p $project -seed $seed -c_lr $lr # -dummy
    done
done