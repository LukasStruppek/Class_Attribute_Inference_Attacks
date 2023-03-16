# Source: https://github.com/google/prompt-to-prompt
#
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, List, Dict
from tqdm.notebook import tqdm
import torch
from inversion import ptp_utils
from inversion.prompt_to_prompt import EmptyControl, AttentionStore


@torch.no_grad()
def text2image_ldm_stable(model,
                          prompt: List[str],
                          controller,
                          num_inference_steps: int = 50,
                          guidance_scale: Optional[float] = 7.5,
                          generator: Optional[torch.Generator] = None,
                          latent: Optional[torch.FloatTensor] = None,
                          uncond_embeddings=None,
                          start_time=50,
                          return_type='image'):
    batch_size = len(prompt)
    ptp_utils.register_attention_control(model, controller)
    height = width = 512

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(
        model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer([""] * batch_size,
                                       padding="max_length",
                                       max_length=max_length,
                                       return_tensors="pt")
        uncond_embeddings_ = model.text_encoder(
            uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings = uncond_embeddings.to(model.device)
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width,
                                            generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        if uncond_embeddings_ is None:
            context = torch.cat([
                uncond_embeddings[i].expand(*text_embeddings.shape),
                text_embeddings
            ])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = ptp_utils.diffusion_step(model,
                                           controller,
                                           latents,
                                           context,
                                           t,
                                           guidance_scale,
                                           low_resource=False)

    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent


def run_and_display(prompts,
                    ldm_model,
                    num_inference_steps,
                    guidance_scale,
                    controller=AttentionStore(),
                    latent=None,
                    run_baseline=False,
                    generator=None,
                    uncond_embeddings=None,
                    verbose=True):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts,
                                         EmptyControl(),
                                         latent=latent,
                                         run_baseline=False,
                                         generator=generator)
        print("with prompt-to-prompt")
    images, x_t = text2image_ldm_stable(
        ldm_model,
        prompts,
        controller,
        latent=latent,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        uncond_embeddings=uncond_embeddings)
    if verbose:
        ptp_utils.view_images(images)
    return images, x_t