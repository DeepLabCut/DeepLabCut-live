import numpy as np
import pytest


class FakeTk:
    def __init__(self):
        self.titles = []
        self.updated = 0
        self.destroyed = False

    def title(self, text):
        self.titles.append(text)

    def update(self):
        self.updated += 1

    def destroy(self):
        self.destroyed = True


class FakeLabel:
    def __init__(self, window):
        self.window = window
        self.packed = False
        self.configured = {}

    def pack(self):
        self.packed = True

    def configure(self, **kwargs):
        self.configured.update(kwargs)


class FakePhotoImage:
    def __init__(self, image=None, master=None):
        self.image = image
        self.master = master


def test_display_init_raises_when_tk_unavailable(monkeypatch):
    import dlclive.display as display_mod

    monkeypatch.setattr(display_mod, "_TKINTER_AVAILABLE", False, raising=False)

    with pytest.raises(ImportError):
        display_mod.Display()


def test_display_frame_creates_window_and_updates(headless_display_env):
    display_mod = headless_display_env
    disp = display_mod.Display(radius=3, pcutoff=0.5)

    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    pose = np.array([[[10, 10, 0.9], [50, 50, 0.2]]])  # 1 animal, 2 bodyparts

    disp.display_frame(frame, pose)

    assert disp.window is not None
    assert disp.lab is not None
    assert disp.lab.packed is True
    assert disp.window.updated == 1
    assert "image" in disp.lab.configured  # configured with PhotoImage


def test_display_draws_only_points_above_cutoff(headless_display_env, monkeypatch):
    display_mod = headless_display_env
    disp = display_mod.Display(radius=3, pcutoff=0.5)

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pose = np.array(
        [
            [
                [10, 10, 0.9],  # draw
                [20, 20, 0.49],  # don't draw
                [30, 30, 0.5001],  # draw (>=)
            ]
        ],
        dtype=float,
    )

    ellipses = []

    class DrawRecorder:
        def ellipse(self, coords, fill=None, outline=None):
            ellipses.append((coords, fill, outline))

    monkeypatch.setattr(display_mod.ImageDraw, "Draw", lambda img: DrawRecorder())

    disp.display_frame(frame, pose)

    assert len(ellipses) == 2


def test_destroy_calls_window_destroy(headless_display_env):
    display_mod = headless_display_env
    disp = display_mod.Display()

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    pose = np.array([[[5, 5, 0.9]]])

    disp.display_frame(frame, pose)
    disp.destroy()

    assert disp.window.destroyed is True


def test_set_display_color_sampling_safe(headless_display_env, monkeypatch):
    display_mod = headless_display_env

    # Provide a fixed colormap list
    class FakeCC:
        bmy = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1)]

    monkeypatch.setattr(display_mod, "cc", FakeCC)

    disp = display_mod.Display(cmap="bmy")
    disp.set_display(im_size=(100, 100), bodyparts=3)

    assert disp.colors is not None
    assert len(disp.colors) >= 3
