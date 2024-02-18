from hashlib import sha512
import os
from io import BytesIO
from typing import List
import time
import requests

import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
import numpy as np

# from lora_diffusion import LoRAManager, monkeypatch_remove_lora
# from t2i_adapters import Adapter
from t2i_adapters import patch_pipe as patch_pipe_t2i_adapter
from PIL import Image

import dotenv
import os

dotenv.load_dotenv()

MODEL_ID = os.environ.get("MODEL_ID", None)
MODEL_CACHE = "diffusers-cache"
SAFETY_MODEL_ID = os.environ.get("SAFETY_MODEL_ID", None)
IS_FP16 = os.environ.get("IS_FP16", "0") == "1"


def url_local_fn(url):
    file_name = sha512(url.encode()).hexdigest() + ".safetensors"
    return file_name


def load_image_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # 确保请求成功
    image_data = BytesIO(response.content)
    image = Image.open(image_data)
    return image


def download_lora(url):
    fn = url_local_fn(url)
    fn_path = f"./{MODEL_CACHE}/lora_models/{fn}"

    # 确保目录存在 如果不存在则创建
    os.makedirs(os.path.dirname(fn_path), exist_ok=True)

    if not os.path.exists(fn_path):
        print("Downloading LoRA model... from", url)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(fn_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    else:
        print("Using disk cache...")

    return fn


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.float16 if IS_FP16 else torch.float32,
        ).to("cuda")

        # patch_pipe_t2i_adapter(self.pipe)

        # self.adapters = {
        #     ext_type: Adapter.from_pretrained(ext_type).to("cuda")
        #     for ext_type, _ in [
        #         ("depth", "antique house"),
        #         ("seg", "motorcycle"),
        #         ("keypose", "elon musk"),
        #         ("sketch", "robot owl"),
        #     ]
        # }

        self.img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.pipe.vae,
            text_encoder=self.pipe.text_encoder,
            tokenizer=self.pipe.tokenizer,
            unet=self.pipe.unet,
            scheduler=self.pipe.scheduler,
            safety_checker=self.pipe.safety_checker,
            feature_extractor=self.pipe.feature_extractor,
        ).to("cuda")

        self.token_size_list: list = []
        self.ranklist: list = []
        self.loaded = None
        self.lora_manager = None

    def set_lora(self, urllists: List[str], scales: List[float]):
        assert len(urllists) == len(scales), "Number of LoRAs and scales must match."

        merged_fn = url_local_fn(f"{'-'.join(urllists)}")

        if self.loaded == merged_fn:
            print("The requested LoRAs are loaded.")
        else:
            # 保存下载后的权重文件名
            lora_weight_files = []

            # 下载每个 LoRA 权重
            for url in urllists:
                fn = download_lora(url)
                lora_weight_files.append(fn)

            # diffusers 的load_lora_weights方法加载权重
            for lora_weight_file, lora_weight_scale in zip(lora_weight_files, scales):
                # self.pipe.load_lora_weights(pretrained_model_name_or_path_or_dict=lora_weight_file)
                # pipeline.load_lora_weights("./models/lora_models/", weight_name="XXX.safetensors")
                # self.pipe.load_lora_weights(f"./{MODEL_CACHE}/lora_models/", weight_name=lora_weight_file)
                self.pipe.load_lora_weights(f"./{MODEL_CACHE}/lora_models/{lora_weight_file}")
                self.pipe.fuse_lora(lora_scale=lora_weight_scale)

            print("LoRA models have been loaded and applied.")
            self.loaded = merged_fn

            # st = time.time()
            # self.lora_manager = LoRAManager(
            #     [download_lora(url) for url in urllists], self.pipe
            # )
            # self.loaded = merged_fn
            # print(f"merging time: {time.time() - st}")

        # self.lora_manager.tune(scales)

    @torch.inference_mode()
    def predict(
            self,
            prompt: str = Input(
                description="Input prompt. The model will generate an image based on this prompt.",
                default="1girl,face,curly hair,red hair,white background",
            ),
            negative_prompt: str = Input(
                description="Specify things to not see in the output",
                default="nsfw",
            ),
            width: int = Input(
                description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
                default=512,
            ),
            height: int = Input(
                description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
                default=512,
            ),
            num_outputs: int = Input(
                description="Number of images to output.",
                ge=1,
                le=4,
                default=1,
            ),
            num_inference_steps: int = Input(
                description="Number of denoising steps.Recommend 25~35", ge=1, le=50, default=30
            ),
            guidance_scale: float = Input(
                description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
            ),
            image: str = Input(
                description="(Img2Img) Inital image to generate variations of. If this is not none, Img2Img will be invoked.Just One",
                default=None,
            ),
            mask: str = Input(
                description="Black and white image to use as mask for inpainting over the image provided. White pixels are inpainted and black pixels are preserved",
                default=None,
            ),
            prompt_strength: float = Input(
                description="(Img2Img) Prompt strength when providing the image. 1.0 corresponds to full destruction of information in init image",
                default=0.8,
            ),
            scheduler: str = Input(
                default="DPMSolverMultistep",
                choices=[
                    "DDIM",
                    "K_EULER",
                    "DPMSolverMultistep",
                    "K_EULER_ANCESTRAL",
                    "PNDM",
                    "KLMS",
                ],
                description="Choose a scheduler.",
            ),
            lora_urls: str = Input(
                description="List of urls for safetensors of lora models, seperated with | .",
                default="",
            ),
            lora_scales: str = Input(
                description="List of scales for safetensors of lora models, seperated with | ",
                default="0.5",
            ),
            seed: int = Input(
                description="Random seed. Leave blank to randomize the seed", default=None
            ),
            # adapter_condition_image: str = Input(
            #     description="(T2I-adapter) Adapter Condition Image to gain extra control over generation. If this is not none, T2I adapter will be invoked. Width, Height of this image must match the above parameter, or dimension of the Img2Img image.",
            #     default=None,
            # ),
            # adapter_type: str = Input(
            #     description="(T2I-adapter) Choose an adapter type for the additional condition.",
            #     choices=["sketch", "seg", "keypose", "depth"],
            #     default="sketch",
            # ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if image is not None:
            pil_image = load_image_from_url(image).convert("RGB")
            width, height = pil_image.size

        print(f"Generating image of {width} x {height} with prompt: {prompt}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        generator = torch.Generator("cuda").manual_seed(seed)

        if len(lora_urls) > 0:
            self.pipe.unload_lora_weights()
            lora_urls = [u.strip() for u in lora_urls.split("|")]
            lora_scales = [float(s.strip()) for s in lora_scales.split("|")]
            self.set_lora(lora_urls, lora_scales)
            # prompt = self.lora_manager.prompt(prompt)
        else:
            print("No LoRA models provided, using default model...")
            # monkeypatch_remove_lora(self.pipe.unet)
            # monkeypatch_remove_lora(self.pipe.text_encoder)
            self.pipe.disable_lora()
            self.pipe.unload_lora_weights()

        # handle t2i adapter
        w_c, h_c = None, None

        # if adapter_condition_image is not None:
        #
        #     cond_img = load_image_from_url(adapter_condition_image)
        #     w_c, h_c = cond_img.size
        #
        #     if w_c != width or h_c != height:
        #         raise ValueError(
        #             "Width and height of the adapter condition image must match the width and height of the generated image."
        #         )
        #
        #     if adapter_type == "sketch":
        #         cond_img = cond_img.convert("L")
        #         cond_img = np.array(cond_img) / 255.0
        #         cond_img = (
        #             torch.from_numpy(cond_img).unsqueeze(0).unsqueeze(0).to("cuda")
        #         )
        #         cond_img = (cond_img > 0.5).float()
        #
        #     else:
        #         cond_img = cond_img.convert("RGB")
        #         cond_img = np.array(cond_img) / 255.0
        #
        #         cond_img = (
        #             torch.from_numpy(cond_img)
        #             .permute(2, 0, 1)
        #             .unsqueeze(0)
        #             .to("cuda")
        #             .float()
        #         )
        #
        #     with torch.no_grad():
        #         adapter_features = self.adapters[adapter_type](cond_img)
        #
        #     self.pipe.unet.set_adapter_features(adapter_features)
        # else:
        #     self.pipe.unet.set_adapter_features(None) # 重置adapter_features

        # either text2img or img2img
        if image is None:
            self.pipe.scheduler = make_scheduler(scheduler, self.pipe.scheduler.config)

            output = self.pipe(
                prompt=[prompt] * num_outputs if prompt is not None else None,
                negative_prompt=[negative_prompt] * num_outputs
                if negative_prompt is not None
                else None,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                generator=generator,
                num_inference_steps=num_inference_steps,
            )
        else:
            # Here is image-to-image
            if mask is not None:
                mask = load_image_from_url(mask).convert("RGB").resize((width, height))
                extra_kwargs = {
                    "mask_image": mask,
                    "image": pil_image,
                    "strength": prompt_strength,
                }
            else:
                mask = None
                extra_kwargs = {"image": pil_image, "strength": prompt_strength}
            self.img2img_pipe.scheduler = make_scheduler(
                scheduler, self.pipe.scheduler.config
            )

            output = self.img2img_pipe(
                prompt=[prompt] * num_outputs if prompt is not None else None,
                negative_prompt=[negative_prompt] * num_outputs
                if negative_prompt is not None
                else None,
                guidance_scale=guidance_scale,
                generator=generator,
                num_inference_steps=num_inference_steps,
                **extra_kwargs,
            )

        output_paths = []
        for i, sample in enumerate(output.images):
            if output.nsfw_content_detected and output.nsfw_content_detected[i]:
                continue

            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                "NSFW content detected. Try running it again, or try a different prompt."
            )

        # 清除所有lora模型
        self.pipe.unload_lora_weights()
        self.loaded = None
        self.lora_manager = None

        return output_paths


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]
