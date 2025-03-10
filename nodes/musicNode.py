import os, sys, base64
import folder_paths
import numpy as np
import torch, random
from comfy.model_management import get_torch_device
from huggingface_hub import snapshot_download
import torchaudio
from scipy.io import wavfile
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from .utils import get_new_counter

modelpath = os.path.join(folder_paths.models_dir, "musicgen")


def init_audio_model(checkpoint):
    audio_processor = AutoProcessor.from_pretrained(checkpoint)
    audio_model = MusicgenForConditionalGeneration.from_pretrained(checkpoint)
    audio_model = audio_model.to(torch.device('cpu'))
    audio_model.generation_config.guidance_scale = 4.0
    audio_model.generation_config.max_new_tokens = 1500
    audio_model.generation_config.temperature = 1.5
    return (audio_processor, audio_model)


class MusicNode:
    def __init__(self):
        self.audio_model = None

    class_type = "MusicNode"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "prompt": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
            "seconds": ("FLOAT", {"default": 5, "min": 1, "max": 1000, "step": 0.1, "display": "number"}),
            "guidance_scale": ("FLOAT", {"default": 4.0, "min": 0, "max": 20}),
            "seed": ("INT", {"default": 0, "min": 0, "max": np.iinfo(np.int32).max}),
            "device": (["auto", "cpu"],),
        }}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "♾️Sound Lab"
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,)

    def run(self, prompt, seconds, guidance_scale, seed, device):
        if seed == -1:
            seed = np.random.randint(0, np.iinfo(np.int32).max)

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        if self.audio_model is None:
            if not os.path.exists(modelpath):
                os.mkdir(modelpath)
            config = os.path.join(modelpath, 'config.json')
            if not os.path.exists(config):
                snapshot_download("facebook/musicgen-small", local_dir=modelpath, endpoint='https://hf-mirror.com')
            self.audio_processor, self.audio_model = init_audio_model(modelpath)

        inputs = self.audio_processor(text=prompt, padding=True, return_tensors="pt")

        if device == 'auto':
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.audio_model.to(torch.device(device))

        tokens_per_second = 1500 / 30
        max_tokens = int(tokens_per_second * seconds)

        sampling_rate = self.audio_model.config.audio_encoder.sampling_rate
        audio_values = self.audio_model.generate(
            **inputs.to(device),
            do_sample=True,
            guidance_scale=guidance_scale,
            max_new_tokens=max_tokens,
        )

        self.audio_model.to(torch.device('cpu'))

        audio = audio_values[0].unsqueeze(0).cpu()

        return ({
                    "waveform": audio,
                    "sample_rate": sampling_rate
                },)


class AudioPlayNode:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = ""
        self.compress_level = 4

    class_type = "AudioPlayNode"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio": ("AUDIO",),
        }}

    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "♾️Mixlab/Audio"
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = ()
    OUTPUT_NODE = True

    def run(self, audio):
        filename_prefix = self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir)
        filename_with_batch_num = filename.replace("%batch_num%", str(1))
        file = f"{filename_with_batch_num}_{counter:05}_.wav"

        waveform = audio['waveform']
        if waveform.ndim == 3:
            waveform = waveform.squeeze(0)
        elif waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        torchaudio.save(
            os.path.join(full_output_folder, file),
            waveform,
            audio["sample_rate"]
        )
        return {"ui": {"audio": [{
            "filename": file,
            "subfolder": subfolder,
            "type": self.type
        }]}}
