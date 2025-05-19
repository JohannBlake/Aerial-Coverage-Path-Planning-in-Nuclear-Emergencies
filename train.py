# Imports
import wandb
import yaml

def main():
    with open("parameters_default.yaml", "r") as file:
        yaml_parameter_defaults_config = yaml.safe_load(file)
    wandb.init(project="Heli-Logs", config=yaml_parameter_defaults_config, settings=wandb.Settings(_disable_stats=True))
    from objective_function import objective
    reward = objective()
    wandb.log({"reward": reward})
if __name__ == "__main__":
    main()