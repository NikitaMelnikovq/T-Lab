from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset

from src.data.transforms import ImageTransform, format_action_prompt
from src.model.action_tokenizer import action_id_to_token


@dataclass
class ExpertRecord:
    episode_id: int
    step_idx: int
    seed: int
    env_id: str
    env_size: int
    mission_text: str
    image_path: str
    action_id: int
    action_token: str
    prompt_variant_id: int


class ExpertStepDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        split: str,
        image_size: int,
        use_prompt_variants: bool = True,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.image_tf = ImageTransform(image_size=image_size)
        self.use_prompt_variants = use_prompt_variants

        metadata_path = self.dataset_dir / "metadata.jsonl"
        split_ids_path = self.dataset_dir / "splits" / f"{split}_ids.txt"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing dataset metadata at {metadata_path}")
        if not split_ids_path.exists():
            raise FileNotFoundError(f"Missing split file at {split_ids_path}")

        with split_ids_path.open("r", encoding="utf-8") as f:
            allowed = {int(x.strip()) for x in f if x.strip()}

        rows: List[ExpertRecord] = []
        with metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if int(row["episode_id"]) not in allowed:
                    continue
                rows.append(
                    ExpertRecord(
                        episode_id=int(row["episode_id"]),
                        step_idx=int(row["step_idx"]),
                        seed=int(row["seed"]),
                        env_id=str(row["env_id"]),
                        env_size=int(row["env_size"]),
                        mission_text=str(row["mission_text"]),
                        image_path=str(row["image_path"]),
                        action_id=int(row["action_id"]),
                        action_token=str(row["action_token"]),
                        prompt_variant_id=int(row.get("prompt_variant_id", 0)),
                    )
                )
        if not rows:
            raise RuntimeError(f"No records loaded for split={split} from {dataset_dir}")

        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        r = self.rows[idx]
        img = Image.open(self.dataset_dir / r.image_path)
        image_tensor = self.image_tf(img)

        variant = r.prompt_variant_id if self.use_prompt_variants else 0
        prompt = format_action_prompt(r.mission_text, variant)

        return {
            "prompt": prompt,
            "mission_text": r.mission_text,
            "image": image_tensor,
            "action_id": r.action_id,
            "action_token": action_id_to_token(r.action_id),
            "episode_id": r.episode_id,
            "step_idx": r.step_idx,
            "seed": r.seed,
        }

    def action_counts(self) -> Dict[int, int]:
        counts: Dict[int, int] = {i: 0 for i in range(7)}
        for r in self.rows:
            counts[int(r.action_id)] += 1
        return counts
