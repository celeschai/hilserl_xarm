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

"""Intervention events of the SpaceMouse teleop, without the device (the reader thread state is set directly)."""

from unittest.mock import PropertyMock, patch

import pytest

pytest.importorskip("pyspacemouse")
pytest.importorskip("pynput")

from lerobot.teleoperators.space_mouse import SpaceMouseTeleop, SpaceMouseTeleopConfig
from lerobot.teleoperators.space_mouse.teleop_space_mouse import GRIPPER_CLOSE, GRIPPER_STAY
from lerobot.teleoperators.utils import TeleopEvents


@pytest.fixture
def make_teleop():
    with patch.object(SpaceMouseTeleop, "is_connected", new_callable=PropertyMock, return_value=True):

        def make(**kwargs) -> SpaceMouseTeleop:
            return SpaceMouseTeleop(SpaceMouseTeleopConfig(**kwargs))

        yield make


def _set_input(teleop: SpaceMouseTeleop, translation=(0.0, 0.0, 0.0), buttons=(0, 0)) -> None:
    teleop._translation[:] = translation
    teleop._buttons = list(buttons)


def _is_intervention(teleop: SpaceMouseTeleop) -> bool:
    teleop.get_action()
    return teleop.get_teleop_events()[TeleopEvents.IS_INTERVENTION]


def test_intervenes_only_while_input(make_teleop):
    teleop = make_teleop()
    assert not _is_intervention(teleop)

    _set_input(teleop, translation=(0.5, 0.0, 0.0))
    assert _is_intervention(teleop)

    _set_input(teleop, translation=(0.0, 0.0, 0.02))  # inside the deadzone
    assert not _is_intervention(teleop)

    _set_input(teleop, buttons=(1, 0))
    assert teleop.get_action()["gripper"] == GRIPPER_CLOSE
    assert teleop.get_teleop_events()[TeleopEvents.IS_INTERVENTION]

    _set_input(teleop)
    assert teleop.get_action()["gripper"] == GRIPPER_STAY
    assert not teleop.get_teleop_events()[TeleopEvents.IS_INTERVENTION]


def test_flag_matches_last_action(make_teleop):
    """Input arriving between `get_action` and `get_teleop_events` must not flag an idle action."""
    teleop = make_teleop()
    teleop.get_action()
    _set_input(teleop, translation=(0.5, 0.0, 0.0))
    assert not teleop.get_teleop_events()[TeleopEvents.IS_INTERVENTION]


def test_latched_intervention_without_input(make_teleop):
    teleop = make_teleop()
    teleop._is_intervention = True
    assert _is_intervention(teleop)


def test_toggle_only(make_teleop):
    teleop = make_teleop(intervene_on_input=False)
    _set_input(teleop, translation=(0.5, 0.0, 0.0), buttons=(1, 0))
    assert not _is_intervention(teleop)
    teleop._is_intervention = True
    assert _is_intervention(teleop)
