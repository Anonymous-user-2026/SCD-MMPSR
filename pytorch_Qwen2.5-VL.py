import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


@dataclass(frozen=True)
class SupportedLLMs:
    QWEN_VL: str = "..."


@dataclass
class Config:
    video_dir: Path = Path("...")
    input_csv: Path = Path("...")
    output_csv: Path = Path("...")
    model_name: str = SupportedLLMs().QWEN_VL
    torch_dtype: torch.dtype = torch.float16
    seed: int = 42
    max_new_tokens: int = 1000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def collect_video_paths(
    video_dir: Path, extensions: tuple[str, ...] = (".mp4",)
) -> dict[str, Path]:
    if not video_dir.is_dir():
        raise ValueError()

    video_paths = {
        video_path.stem: video_path
        for video_path in video_dir.rglob("*")
        if video_path.is_file() and video_path.suffix.lower() in extensions
    }

    return video_paths


PROMPT_TEMPLATE: str = """
You are an expert in visual human behavior analysis. Carefully examine the provided video clip, which features a person facing the camera. Your task is to describe, in continuous natural language, the person’s visible emotional state, personality tendencies, or possible signs of ambivalence and hesitancy as reflected through their nonverbal behavior.

Focus exclusively on observable cues such as facial muscle movements (eyes, eyebrows, mouth, gaze), body posture, gestures, and head motions. Infer emotional tendencies (neutral, anger, disgust, fear, happiness, sadness, surprise), personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism), or subtle conflicting signals of uncertainty and hesitation when visible.

In your description:
- Comment on the person’s appearance, posture, gestures, and expressiveness as indicators of emotional state, personality, or ambivalence.
- Observe and explain facial expressions and body movements as cues, highlighting consistency or discordance across behaviors.
- Avoid assumptions about personal background, spoken content, or context beyond what is visually observable.
- If the state appears mixed or ambiguous, briefly mention this with a short explanation based on visible cues.

Your final response must be a fluent, continuous natural language interpretation of the person’s visible behavior in the video, written as a single coherent paragraph without any line breaks, bullet points, special characters, or formatting. The response must express a complete, finished thought and must not exceed 75 tokens in total.
"""


@dataclass
class InferenceEngine:
    config: Config
    model: Qwen2_5_VLForConditionalGeneration = None
    processor: AutoProcessor = None

    def __post_init__(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_name,
            torch_dtype=self.config.torch_dtype,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)

    def build_messages(self, video_path: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": PROMPT_TEMPLATE},
                ],
            }
        ]

    def process_video(self, video_path: str) -> str:
        messages = self.build_messages(video_path)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True
        )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs, max_new_tokens=self.config.max_new_tokens
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text.strip()


def main() -> None:
    config = Config()
    set_seed(config.seed)

    engine = InferenceEngine(config)
    df = pl.read_csv(config.input_csv)

    video_paths = collect_video_paths(config.video_dir)

    if "text_llm" not in df.columns:
        df = df.with_columns(pl.lit("").alias("text_llm"))

    if df.height == 0:
        return

    total = df.height

    updated_rows: list[dict] = []

    pbar = tqdm(enumerate(df.iter_rows(named=True)), total=total)
    for idx, row in pbar:
        video_name = row["video_name"]
        video_path = video_paths.get(video_name)

        if not video_path:
            continue

        try:
            result_text = engine.process_video(str(video_path))
            row["text_llm"] = result_text
        except Exception as e:
            pbar.set_postfix(idx=idx, e=e, video=video_path)
            continue

        updated_rows.append(row)
        pl.DataFrame(updated_rows).write_csv(config.output_csv)


if __name__ == "__main__":
    main()
