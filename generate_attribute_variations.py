import argparse
import os

import torch
import torchvision.transforms as T
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image

from inversion.inference import run_and_display
from inversion.prompt_to_prompt import make_controller
from utils.synthesizing_config_parser import SynthesizingConfigParser


def main():
    # Define and parse arguments
    parser = create_parser()
    config, args = parse_arguments(parser)

    latents = sorted(os.listdir(config.image_latents))

    rtpt = config.create_rtpt(len(latents))
    rtpt.start()

    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    ldm_model = load_ldm_model(device, config.ldm_model_path)
    filter_model = config.load_filter_model().to(device)
    filter_model.eval()
    filter_transformations = config.create_filter_transformations()

    attributes = config.attributes

    # create output folder
    create_output_folder(config.output_folder, config.attribute_class,
                         attributes)

    img_counter = 0
    for file in latents:
        # load latent vectors, embeddings and other information
        data = torch.load(os.path.join(config.image_latents, file))

        # create prompts
        standard_prompt = ''.join(data['prompt'])
        prompts = create_prompts(standard_prompt, attributes)
        prompts = [standard_prompt] + prompts

        controller = make_controller(prompts, False,
                                     config.cross_replace_steps,
                                     config.self_replace_steps,
                                     ldm_model.tokenizer, device, None, None)

        # generate images
        images, _ = run_and_display(
            prompts,
            ldm_model,
            data['num_inference_steps'],
            data['guidance_scale'],
            controller,
            run_baseline=False,
            latent=data['latents'],
            uncond_embeddings=data['uncond_embeddings'])

        # filter images
        with torch.no_grad():
            images_tensor = torch.from_numpy(images[1:]).permute(
                0, 3, 1, 2).to(device) / 255.0
            images_trans = filter_transformations(images_tensor)
            predictions = filter_model(
                images_trans)[:, :len(attributes)].softmax(dim=1)
            scores = predictions.diag()
            labels = torch.argmax(predictions, dim=1)

            if torch.all(labels == torch.tensor(
                [i for i in range(len(labels))], device=device)) == False:
                rtpt.step()
                continue
            if torch.all(scores > config.threshold) == False:
                rtpt.step()
                continue

        # save images
        for img, attribute in zip(images[1:], attributes):
            img = Image.fromarray(img)
            file_name = file.split('.')[0] + '.jpg'
            if attribute is None:
                attribute = 'standard'
            img.save(
                os.path.join(config.output_folder, config.attribute_class,
                             attribute.replace(' ', '_').replace(',', '_'),
                             file_name))

        img_counter += 1
        print(f'{img_counter} of {config.num_samples} generated')
        if img_counter >= config.num_samples:
            break

        rtpt.step()


def load_ldm_model(device, model_path):
    scheduler = DDIMScheduler(beta_start=0.00085,
                              beta_end=0.012,
                              beta_schedule="scaled_linear",
                              clip_sample=False,
                              set_alpha_to_one=False)
    ldm_model = StableDiffusionPipeline.from_pretrained(
        model_path,
        cache_dir='./weights',
        use_auth_token="",
        scheduler=scheduler).to(device)

    return ldm_model


def create_output_folder(output_folder, attribute_class, attributes):
    for attribute in attributes:
        if attribute is None:
            attribute = 'standard'
        attribute = attribute.replace(' ', '_')
        attribute_class = attribute_class.replace(' ', '_')
        folder = os.path.join(output_folder, attribute_class, attribute)
        os.makedirs(folder, exist_ok=True)


def create_prompts(standard_prompt, attributes):
    prompts = []
    for attribute in attributes:
        if attribute is not None:
            prompts.append(standard_prompt + ', ' + attribute)
        else:
            prompts.append(standard_prompt)
    return prompts


def create_parser():
    parser = argparse.ArgumentParser(
        description='Performing model inversion attack')
    parser.add_argument('-c',
                        '--config',
                        default=None,
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')
    return parser


def parse_arguments(parser):
    args = parser.parse_args()

    if not args.config:
        print(
            "Configuration file is missing. Please check the provided path. Execution is stopped."
        )
        exit()

    # Load synthesizing config
    config = SynthesizingConfigParser(args.config)

    return config, args


if __name__ == '__main__':
    main()
