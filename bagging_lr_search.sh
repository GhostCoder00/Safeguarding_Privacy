printf "\nRunning bagging learning rate search\n"

for project in colorado korea engagenet daisee germany
do
    for lr in -5 -4.5 -4 -3.5 -3 -2.5 -2 -1.5
    do
        for seed in 0 1 2 3 4
        do
            python main_bagging.py -p $project -seed $seed -c_lr $lr # -hard # -same
        done
    done
done