import torch
from models.classifier import Classifier

import wandb


def load_model(run_path,
               model_path=None,
               config=None,
               architecture=None,
               num_classes=None,
               replace=True):

    # Get file path at wandb if not set
    if model_path is None:
        api = wandb.Api(timeout=60)
        run = api.run(run_path)
        model_path = run.config["model_path"]
        architecture = run.config['Architecture']

    # Create model
    if num_classes is None:
        num_classes = run.config["num_classes"]

    if config:
        model = config.create_model()
    elif architecture is None:
        architecture = model_path.split('/')[-1].split('_')[0]

    model = Classifier(num_classes, in_channels=3, architecture=architecture)
    model.num_classes = num_classes

    # Load weights from wandb
    file_model = wandb.restore(model_path,
                               run_path=run_path,
                               root='./weights',
                               replace=replace)

    # Change keys from compiled model weights
    weights = torch.load(file_model.name,
                         map_location='cpu')['model_state_dict']
    weights = {k.replace('_orig_mod.', ''): v for k, v in weights.items()}
                 
    model.load_state_dict(weights)
    model.wandb_name = run.name
    model.eval()
    return model


def init_wandb_logging(target_model_name, target_model_architecture, config,
                       args):
    if not 'name' in config.wandb['wandb_init_args']:
        config.wandb['wandb_init_args'][
            'name'] = f'{target_model_name}_{config.attack["attribute"]}'
    wandb_config = config.create_wandb_config()
    wandb_config['architecture'] = target_model_architecture
    run = wandb.init(config=wandb_config, **config.wandb['wandb_init_args'])
    wandb.save(args.config)
    return run
