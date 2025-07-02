printf "\nRunning Germany FL learning rate search\n"

project=germany

for c_lr in -1.5 -1 # -5 -4.5 -4 -3.5 -3 -2.5 -2
do
    for seed in 0 1 2 3 4
    do
        python main_FL.py -p $project -seed $seed -c_lr $c_lr -fl FedAvg # -dummy
    done
done