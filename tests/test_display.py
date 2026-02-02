from unittest.mock import ANY, MagicMock

import numpy as np
import pytest


def test_display_init_raises_when_tk_unavailable(monkeypatch):
    import dlclive.display as display_mod

    monkeypatch.setattr(display_mod, "_TKINTER_AVAILABLE", False, raising=False)

    with pytest.raises(ImportError):
        display_mod.Display()


def test_display_frame_creates_window_and_updates(headless_display_env):
    env = headless_display_env
    display_mod = env.mod
    disp = display_mod.Display(radius=3, pcutoff=0.5)

    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    pose = np.array([[[10, 10, 0.9], [50, 50, 0.2]]])  # 1 animal, 2 bodyparts

    disp.display_frame(frame, pose)

    # Window created and initialized
    env.tk_ctor.assert_called_once_with()
    env.tk.title.assert_called_once_with("DLC Live")

    # Label created and packed
    env.label_ctor.assert_called_once_with(env.tk)
    env.label.pack.assert_called_once()

    # PhotoImage created with correct master + image passed
    env.photo_ctor.assert_called_once_with(image=ANY, master=env.tk)

    # Image configured on label and window updated
    env.label.configure.assert_called_once_with(image=env.photo)
    env.tk.update.assert_called_once_with()


def test_display_draws_only_points_above_cutoff(headless_display_env, monkeypatch):
    env = headless_display_env
    display_mod = env.mod
    disp = display_mod.Display(radius=3, pcutoff=0.5)

    # Patch colormap so color sampling is deterministic and always long enough
    class FakeCC:
        bmy = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]

    monkeypatch.setattr(display_mod, "cc", FakeCC)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pose = np.array(
        [
            [
                [10, 10, 0.9],  # draw
                [20, 20, 0.49],  # don't draw
                [30, 30, 0.5001],  # draw (> pcutoff)
            ]
        ],
        dtype=float,
    )

    draw = MagicMock(name="DrawInstance")
    monkeypatch.setattr(display_mod.ImageDraw, "Draw", MagicMock(return_value=draw))

    disp.display_frame(frame, pose)

    # Two points above cutoff => two ellipse calls
    assert draw.ellipse.call_count == 2


def test_destroy_calls_window_destroy(headless_display_env):
    env = headless_display_env
    display_mod = env.mod
    disp = display_mod.Display()

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    pose = np.array([[[5, 5, 0.9]]])

    disp.display_frame(frame, pose)
    disp.destroy()

    env.tk.destroy.assert_called_once_with()


def test_set_display_color_sampling_safe(headless_display_env, monkeypatch):
    env = headless_display_env
    display_mod = env.mod

    class FakeCC:
        bmy = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1)]

    monkeypatch.setattr(display_mod, "cc", FakeCC)

    disp = display_mod.Display(cmap="bmy")
    disp.set_display(im_size=(100, 100), bodyparts=3)

    assert disp.colors is not None
    assert len(disp.colors) >= 3

    # Also verify window setup calls happened
    env.tk_ctor.assert_called_once_with()
    env.tk.title.assert_called_once_with("DLC Live")
    env.label_ctor.assert_called_once_with(env.tk)
    env.label.pack.assert_called_once()
