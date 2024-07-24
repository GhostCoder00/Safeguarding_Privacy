for seed in 0 1 2 3 4
do
    project=colorado
    c_lr=-3
    fedadam_lr=-4.5
    fedaws_lr=-3.5
    turbosvm_lr=-3
    python main_FL.py -p $project -seed $seed -c_lr $c_lr -fl FedAvg
    python main_FL.py -p $project -seed $seed -c_lr $c_lr -fl FedProx
    python main_FL.py -p $project -seed $seed -c_lr $c_lr -fl MOON
    python main_FL.py -p $project -seed $seed -c_lr $c_lr -fl FedAdam -g_lr $fedadam_lr
    python main_FL.py -p $project -seed $seed -c_lr $c_lr -fl FedAwS -l_lr $fedaws_lr
    python main_FL.py -p $project -seed $seed -c_lr $c_lr -fl TurboSVM -l_lr $turbosvm_lr

    project=korea
    c_lr=-2
    fedadam_lr=-3.5
    fedaws_lr=-2
    turbosvm_lr=-3.5
    python main_FL.py -p $project -seed $seed -c_lr $c_lr -fl FedAvg
    python main_FL.py -p $project -seed $seed -c_lr $c_lr -fl FedProx
    python main_FL.py -p $project -seed $seed -c_lr $c_lr -fl MOON
    python main_FL.py -p $project -seed $seed -c_lr $c_lr -fl FedAdam -g_lr $fedadam_lr
    python main_FL.py -p $project -seed $seed -c_lr $c_lr -fl FedAwS -l_lr $fedaws_lr
    python main_FL.py -p $project -seed $seed -c_lr $c_lr -fl TurboSVM -l_lr $turbosvm_lr

    project=engagenet
    c_lr=-3.5
    fedadam_lr=-4.5
    fedaws_lr=-3.5
    turbosvm_lr=-3
    python main_FL.py -p $project -seed $seed -c_lr $c_lr -fl FedAvg
    python main_FL.py -p $project -seed $seed -c_lr $c_lr -fl FedProx
    python main_FL.py -p $project -seed $seed -c_lr $c_lr -fl MOON
    python main_FL.py -p $project -seed $seed -c_lr $c_lr -fl FedAdam -g_lr $fedadam_lr
    python main_FL.py -p $project -seed $seed -c_lr $c_lr -fl FedAwS -l_lr $fedaws_lr
    python main_FL.py -p $project -seed $seed -c_lr $c_lr -fl TurboSVM -l_lr $turbosvm_lr

    project=daisee
    c_lr=-3
    fedadam_lr=-5
    fedaws_lr=-5
    turbosvm_lr=-5
    python main_FL.py -p $project -seed $seed -c_lr $c_lr -fl FedAvg
    python main_FL.py -p $project -seed $seed -c_lr $c_lr -fl FedProx
    python main_FL.py -p $project -seed $seed -c_lr $c_lr -fl MOON
    python main_FL.py -p $project -seed $seed -c_lr $c_lr -fl FedAdam -g_lr $fedadam_lr
    python main_FL.py -p $project -seed $seed -c_lr $c_lr -fl FedAwS -l_lr $fedaws_lr
    python main_FL.py -p $project -seed $seed -c_lr $c_lr -fl TurboSVM -l_lr $turbosvm_lr
done