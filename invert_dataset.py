import argparse
import os

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from rtpt import RTPT
from tqdm import tqdm

from inversion.null_text_inversion import NullInversion


def main():
    parser = create_parser()
    args = parser.parse_args()

    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    image_paths = sorted(os.listdir(args.image_folder))
    os.makedirs(args.output_folder, exist_ok=True)

    rtpt = RTPT(args.user, 'Null-Text Inversion', len(image_paths))
    rtpt.start()

    ldm_model = load_ldm_model(device)

    null_inversion = NullInversion(ldm_model,
                                   num_ddim_steps=50,
                                   guidance_scale=7.5)

    for input_file in tqdm(image_paths):
        _, latent, uncond_embeddings = null_inversion.invert(os.path.join(
            args.image_folder, input_file),
                                                             args.prompt,
                                                             verbose=False)
        output_file = input_file.split('.')[0] + '.pt'
        torch.save(
            {
                'prompt': args.prompt,
                'guidance_scale': 7.5,
                'num_inference_steps': 50,
                'latents': latent.cpu(),
                'uncond_embeddings': uncond_embeddings.cpu()
            }, os.path.join(args.output_folder, output_file))
        rtpt.step()


def load_ldm_model(device):
    scheduler = DDIMScheduler(beta_start=0.00085,
                              beta_end=0.012,
                              beta_schedule="scaled_linear",
                              clip_sample=False,
                              set_alpha_to_one=False)
    ldm_model = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        cache_dir='./weights',
        use_auth_token="",
        scheduler=scheduler).to(device)

    return ldm_model


def create_parser():
    parser = argparse.ArgumentParser(
        description='Performing model inversion attack')
    parser.add_argument('-i',
                        '--image_folder',
                        default=None,
                        type=str,
                        dest="image_folder",
                        help='Input image folder')
    parser.add_argument('-o',
                        '--output_folder',
                        default=None,
                        type=str,
                        dest="output_folder",
                        help='Output folder for inverted images')
    parser.add_argument('-p',
                        '--prompt',
                        default='a photo of a person',
                        type=str,
                        dest="prompt",
                        help='Prompt for null-text inversion')
    parser.add_argument('-u',
                        '--user',
                        default='XX',
                        type=str,
                        dest="user",
                        help='User initials for RTPT')

    return parser


if __name__ == '__main__':
    main()
