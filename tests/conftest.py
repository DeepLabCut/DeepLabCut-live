import pytest


@pytest.fixture
def headless_display_env(monkeypatch):
    # Import module under test
    from test_display import FakeLabel, FakePhotoImage, FakeTk

    import dlclive.display as display_mod

    # Force tkinter availability and patch UI components
    monkeypatch.setattr(display_mod, "_TKINTER_AVAILABLE", True, raising=False)
    monkeypatch.setattr(display_mod, "Tk", FakeTk, raising=False)
    monkeypatch.setattr(display_mod, "Label", FakeLabel, raising=False)

    # Patch ImageTk.PhotoImage
    class FakeImageTkModule:
        PhotoImage = FakePhotoImage

    monkeypatch.setattr(display_mod, "ImageTk", FakeImageTkModule, raising=False)

    return display_mod
