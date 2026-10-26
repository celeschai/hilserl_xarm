#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from lerobot.utils.constants import OBS_IMAGE
from lerobot.utils.transition import Transition

JOURNAL_DIR = "dataset_journal"


class TransitionJournal:
    """
    Append-only journal of the transitions received since the replay buffers were last saved in full.

    Saving the full replay buffers re-encodes every frame and takes minutes, during which the learner does not
    train. As in rail-berkeley/hil-serl, whose actor appends its transitions to a file every `buffer_period`
    steps, each batch of transitions is instead written to its own small file as it arrives. The full buffers
    are saved on shutdown, and the journal is replayed on top of the last full save when the buffers are
    loaded, so a crash loses at most the batch being written.

    Images are stored as uint8 and next states are left out: the replay buffers take the next state from the
    next stored step.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._next_index: int | None = None

    def _files(self) -> list[Path]:
        return sorted(self.directory.glob("*.pt")) if self.directory.is_dir() else []

    def append(self, transitions: list[Transition]) -> None:
        if self._next_index is None:
            files = self._files()
            self._next_index = int(files[-1].stem) + 1 if files else 0
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{self._next_index:08d}.pt"
        tmp_path = path.with_suffix(".tmp")
        torch.save([_compact(t) for t in transitions], tmp_path)
        os.replace(tmp_path, path)
        self._next_index += 1

    def read(self) -> Iterator[Transition]:
        """The journaled transitions, in the order they were received."""
        for path in self._files():
            for transition in torch.load(path, weights_only=True):
                yield _expand(transition)

    def clear(self) -> None:
        """Forget the journaled transitions, once they are in a full save of the buffers."""
        for path in self._files():
            path.unlink()
        self._next_index = 0


def _compact(transition: Transition) -> dict[str, Any]:
    state = {
        key: (value.clamp(0, 1) * 255).round().to(torch.uint8)
        if key.startswith(OBS_IMAGE) and value.is_floating_point()
        else value
        for key, value in transition["state"].items()
    }
    return {
        "state": state,
        "action": transition["action"],
        "reward": transition["reward"],
        "done": transition["done"],
        "truncated": transition["truncated"],
        "complementary_info": transition.get("complementary_info"),
    }


def _expand(compact: dict[str, Any]) -> Transition:
    state = {
        key: value.float() / 255 if value.dtype == torch.uint8 else value
        for key, value in compact["state"].items()
    }
    return Transition(
        state=state,
        action=compact["action"],
        reward=compact["reward"],
        next_state=state,
        done=compact["done"],
        truncated=compact["truncated"],
        complementary_info=compact["complementary_info"],
    )
