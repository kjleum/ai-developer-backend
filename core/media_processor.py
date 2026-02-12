import os
import base64
import aiohttp
from typing import Dict, Any, Optional, List, Union
import json

class MediaProcessor:
    """Обработка медиа: изображения, видео, аудио, TTS, STT"""

    def __init__(self):
        # Stable Diffusion (Automatic1111 API)
        self.sd_url = os.getenv("SD_API_URL", "http://localhost:7860")
        self.sd_available = False

        # ComfyUI
        self.comfy_url = os.getenv("COMFYUI_URL", "http://localhost:8188")

        # Kandinsky API (FusionBrain или локальный)
        self.kandinsky_url = os.getenv("KANDINSKY_URL", "http://localhost:8001")

        # TTS/STT
        self.coqui_url = os.getenv("COQUI_TTS_URL", "http://localhost:5002")
        self.whisper_url = os.getenv("WHISPER_URL", "http://localhost:9000")

        # Video generation
        self.wan_url = os.getenv("WAN_API_URL", "http://localhost:8080")

        # Ollama для локальной генерации
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")

    async def check_services(self) -> Dict[str, bool]:
        """Проверить доступность всех сервисов"""
        services = {}

        async with aiohttp.ClientSession() as session:
            # Stable Diffusion
            try:
                async with session.get(f"{self.sd_url}/sdapi/v1/samplers", timeout=aiohttp.ClientTimeout(total=2)):
                    services["stable_diffusion"] = True
            except:
                services["stable_diffusion"] = False

            # ComfyUI
            try:
                async with session.get(f"{self.comfy_url}/system_stats", timeout=aiohttp.ClientTimeout(total=2)):
                    services["comfyui"] = True
            except:
                services["comfyui"] = False

            # Ollama (для локальных моделей)
            try:
                async with session.get(f"{self.ollama_url}/api/tags", timeout=aiohttp.ClientTimeout(total=2)):
                    services["ollama"] = True
            except:
                services["ollama"] = False

            # TTS
            try:
                async with session.get(f"{self.coqui_url}/", timeout=aiohttp.ClientTimeout(total=2)):
                    services["tts"] = True
            except:
                services["tts"] = False

            # Whisper
            try:
                async with session.get(f"{self.whisper_url}/", timeout=aiohttp.ClientTimeout(total=2)):
                    services["stt"] = True
            except:
                services["stt"] = False

        return services

    # === ИЗОБРАЖЕНИЯ ===

    async def generate_image(
        self, 
        prompt: str, 
        negative_prompt: str = "",
        width: int = 512, 
        height: int = 512,
        steps: int = 30,
        cfg_scale: float = 7.0,
        sampler: str = "DPM++ 2M Karras",
        model: str = "sd",
        seed: int = -1,
        **kwargs
    ) -> Dict[str, Any]:
        """Генерация изображения через Stable Diffusion или Kandinsky"""

        if model == "kandinsky":
            return await self._generate_kandinsky(prompt, width, height, **kwargs)

        # Stable Diffusion через Automatic1111
        async with aiohttp.ClientSession() as session:
            payload = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "sampler_name": sampler,
                "seed": seed,
                "batch_size": 1,
                "n_iter": 1,
                "save_images": False,
                "send_images": True
            }

            # Добавляем ControlNet если указан
            if "controlnet" in kwargs:
                payload["alwayson_scripts"] = {
                    "ControlNet": {
                        "args": [{
                            "input_image": kwargs["controlnet"]["image"],
                            "model": kwargs["controlnet"].get("model", "control_v11p_sd15_canny"),
                            "module": kwargs["controlnet"].get("preprocessor", "canny"),
                            "weight": kwargs["controlnet"].get("weight", 1.0)
                        }]
                    }
                }

            async with session.post(
                f"{self.sd_url}/sdapi/v1/txt2img",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"SD error: {await resp.text()}")

                data = await resp.json()
                images = data.get("images", [])

                if not images:
                    raise Exception("No images generated")

                return {
                    "success": True,
                    "images": images,  # base64 encoded
                    "parameters": data.get("parameters", {}),
                    "info": json.loads(data.get("info", "{}"))
                }

    async def _generate_kandinsky(self, prompt: str, width: int, height: int, **kwargs) -> Dict[str, Any]:
        """Генерация через Kandinsky 3.0/4.0"""
        # Используем FusionBrain API или локальный сервер
        async with aiohttp.ClientSession() as session:
            payload = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": kwargs.get("steps", 50),
                "guidance_scale": kwargs.get("cfg_scale", 7.5)
            }

            async with session.post(
                f"{self.kandinsky_url}/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=180)
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"Kandinsky error: {await resp.text()}")

                data = await resp.json()
                return {
                    "success": True,
                    "images": [data.get("image")],
                    "model": "kandinsky-4"
                }

    async def image_to_image(
        self,
        image: str,  # base64
        prompt: str,
        strength: float = 0.75,
        **kwargs
    ) -> Dict[str, Any]:
        """Img2Img преобразование"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "init_images": [image],
                "prompt": prompt,
                "strength": strength,
                "steps": kwargs.get("steps", 30),
                "cfg_scale": kwargs.get("cfg_scale", 7.0)
            }

            async with session.post(
                f"{self.sd_url}/sdapi/v1/img2img",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                data = await resp.json()
                return {
                    "success": True,
                    "images": data.get("images", [])
                }

    async def inpaint(
        self,
        image: str,
        mask: str,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Inpainting - замена части изображения"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "init_images": [image],
                "mask": mask,
                "prompt": prompt,
                "steps": kwargs.get("steps", 30),
                "cfg_scale": kwargs.get("cfg_scale", 7.0),
                "denoising_strength": kwargs.get("strength", 0.8)
            }

            async with session.post(
                f"{self.sd_url}/sdapi/v1/img2img",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                data = await resp.json()
                return {
                    "success": True,
                    "images": data.get("images", [])
                }

    async def upscale(self, image: str, scale: int = 2) -> Dict[str, Any]:
        """Увеличение разрешения"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "image": image,
                "upscaler_1": "R-ESRGAN 4x+",
                "upscaling_resize": scale
            }

            async with session.post(
                f"{self.sd_url}/sdapi/v1/extra-single-image",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                data = await resp.json()
                return {
                    "success": True,
                    "image": data.get("image")
                }

    # === ВИДЕО ===

    async def generate_video(
        self,
        prompt: str,
        image: Optional[str] = None,  # Для image-to-video
        duration: int = 4,
        fps: int = 8,
        resolution: str = "512x512",
        model: str = "wan"  # wan или kandinsky
    ) -> Dict[str, Any]:
        """Генерация видео из текста или изображения"""

        if model == "kandinsky":
            return await self._generate_video_kandinsky(prompt, image, duration, resolution)

        # Wan-2.1 (Wan-AI)
        async with aiohttp.ClientSession() as session:
            payload = {
                "prompt": prompt,
                "image": image,  # Для I2V
                "video_length": duration * fps,
                "fps": fps,
                "resolution": resolution,
                "seed": -1
            }

            async with session.post(
                f"{self.wan_url}/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=600)  # 10 минут
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"Wan error: {await resp.text()}")

                data = await resp.json()
                return {
                    "success": True,
                    "video": data.get("video"),  # base64 или URL
                    "duration": duration,
                    "fps": fps,
                    "model": "wan-2.1"
                }

    async def _generate_video_kandinsky(
        self, 
        prompt: str, 
        image: Optional[str],
        duration: int,
        resolution: str
    ) -> Dict[str, Any]:
        """Kandinsky Video Lite"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "prompt": prompt,
                "init_image": image,
                "num_frames": duration * 8,
                "fps": 8,
                "motion_bucket_id": 127
            }

            async with session.post(
                f"{self.kandinsky_url}/video",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as resp:
                data = await resp.json()
                return {
                    "success": True,
                    "video": data.get("video"),
                    "model": "kandinsky-video-lite"
                }

    # === АУДИО ===

    async def text_to_speech(
        self,
        text: str,
        voice: str = "default",
        language: str = "ru",
        speed: float = 1.0,
        emotion: str = "neutral"
    ) -> Dict[str, Any]:
        """Текст в речь (Coqui TTS, Silero, или Piper)"""

        async with aiohttp.ClientSession() as session:
            payload = {
                "text": text,
                "speaker_id": voice,
                "language": language,
                "speed": speed
            }

            async with session.post(
                f"{self.coqui_url}/api/tts",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    # Fallback на Silero через Ollama
                    return await self._tts_ollama(text, voice)

                # Получаем аудио
                audio_data = await resp.read()
                return {
                    "success": True,
                    "audio": base64.b64encode(audio_data).decode(),
                    "format": "wav",
                    "duration": len(text) * 0.1  # Примерная оценка
                }

    async def _tts_ollama(self, text: str, voice: str) -> Dict[str, Any]:
        """Fallback TTS через Ollama (если есть соответствующая модель)"""
        # Заглушка - в реальности нужна модель типа melotts или类似
        return {
            "success": False,
            "error": "TTS service not available",
            "fallback": "Please install Coqui TTS: docker run -p 5002:5002 coqui/tts"
        }

    async def speech_to_text(
        self,
        audio: str,  # base64
        language: str = "ru",
        model: str = "whisper"
    ) -> Dict[str, Any]:
        """Речь в текст (Whisper или Vosk)"""

        if model == "whisper":
            async with aiohttp.ClientSession() as session:
                # Декодируем base64
                audio_bytes = base64.b64decode(audio)

                data = aiohttp.FormData()
                data.add_field('audio', audio_bytes, filename='audio.wav', content_type='audio/wav')
                data.add_field('language', language)

                async with session.post(
                    f"{self.whisper_url}/transcribe",
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    result = await resp.json()
                    return {
                        "success": True,
                        "text": result.get("text", ""),
                        "segments": result.get("segments", []),
                        "language": result.get("language", language)
                    }

        elif model == "vosk":
            # Vosk API
            pass

        return {"success": False, "error": "Unknown STT model"}

    async def clone_voice(
        self,
        audio_samples: List[str],  # base64 encoded audio samples
        name: str
    ) -> Dict[str, Any]:
        """Клонирование голоса (Coqui TTS или类似)"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "samples": audio_samples,
                "name": name
            }

            async with session.post(
                f"{self.coqui_url}/api/clone-voice",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                data = await resp.json()
                return {
                    "success": True,
                    "voice_id": data.get("voice_id"),
                    "name": name
                }

    # === BATCH PROCESSING ===

    async def batch_generate_images(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Пакетная генерация изображений"""
        tasks = [self.generate_image(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def batch_tts(
        self,
        texts: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Пакетная генерация речи"""
        tasks = [self.text_to_speech(text, **kwargs) for text in texts]
        return await asyncio.gather(*tasks, return_exceptions=True)
