import os
from itertools import product


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))


datasets = ["cora", "facebook", "lastfm", "pubmed"]
x_epsilons = [0.1]
y_epsilons = [1]
x_steps = list(range(17))
y_steps = list(range(17))
learning_rates = [0.01, 0.001, 0.0001]
models = ["sage"]
weight_decays = [0.01, 0.001, 0.0001, 0]
dropouts = [0, 0.25, 0.5, 0.75]

for dataset in datasets:
    cmd_list = []
    default_options = " --dataset " + dataset + "  --feature raw    --mechanism mbm  --forward_correction True  -s 12345  -r " \
                                                "10  -o ./results "
    configs = product_dict(
        model=models,
        x_eps=x_epsilons,
        y_eps=y_epsilons,
        x_steps=x_steps,
        learning_rate=learning_rates,
        weight_decay=weight_decays,
        dropout=dropouts,
    )

    for config in configs:
        config["y_steps"] = config["x_steps"]
        options = ' '.join([f' --{param} {value} ' for param, value in config.items()])
        command = f'python main.py {options} {default_options}'
        cmd_list.append(command)

    with open(os.path.join("./jobs/", dataset + '-' + models[0]  + '-' + str(x_epsilons[0]) + '.jobs'), 'w') as file:
        for cmd in cmd_list:
            file.write(cmd + '\n')
