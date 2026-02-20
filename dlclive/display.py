"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""

try:
    from tkinter import Label, Tk

    from PIL import ImageTk

    _TKINTER_AVAILABLE = True
except ImportError:
    _TKINTER_AVAILABLE = False
    Label = None
    Tk = None
    ImageTk = None

import colorcet as cc
from PIL import Image, ImageDraw


class Display:
    """
    Simple object to display frames with DLC labels.

    Parameters
    -----------
    cmap: string
        The Matplotlib colormap to use.
    pcutoff : float
        likelihood threshold to display points
    """

    def __init__(self, cmap="bmy", radius=3, pcutoff=0.5):
        if not _TKINTER_AVAILABLE:
            raise ImportError("tkinter is not available. Display functionality requires tkinter. ")
        self.cmap = cmap
        self.colors = None
        self.radius = radius
        self.pcutoff = pcutoff
        self.window = None

    def set_display(self, im_size, bodyparts):
        """Create tkinter window to display image

        Parameters
        ----------
        im_size : tuple
            (width, height) of image
        bodyparts : int
            number of bodyparts
        """

        self.window = Tk()
        self.window.title("DLC Live")
        self.lab = Label(self.window)
        self.lab.pack()

        all_colors = getattr(cc, self.cmap)
        # Avoid 0 step
        step = max(1, int(len(all_colors) / bodyparts))
        self.colors = all_colors[::step]

    def display_frame(self, frame, pose=None):
        """
        Display the image with DeepLabCut labels using opencv imshow

        Parameters
        -----------
        frame :class:`numpy.ndarray`
            an image as a numpy array

        pose :class:`numpy.ndarray`
            the pose estimated by DeepLabCut for the image
        """
        if not _TKINTER_AVAILABLE:
            raise ImportError("tkinter is not available. Cannot display frames.")

        im_size = (frame.shape[1], frame.shape[0])
        img = Image.fromarray(frame)  # avoid undefined image if pose is None
        if pose is not None:
            draw = ImageDraw.Draw(img)

            if len(pose.shape) == 2:
                pose = pose[None]

            if self.window is None:
                self.set_display(im_size=im_size, bodyparts=pose.shape[1])

            for i in range(pose.shape[0]):
                for j in range(pose.shape[1]):
                    if pose[i, j, 2] > self.pcutoff:
                        try:
                            x0 = max(0, pose[i, j, 0] - self.radius)
                            x1 = min(im_size[0], pose[i, j, 0] + self.radius)
                            y0 = max(0, pose[i, j, 1] - self.radius)
                            y1 = min(im_size[1], pose[i, j, 1] + self.radius)
                            coords = [x0, y0, x1, y1]
                            draw.ellipse(coords, fill=self.colors[j], outline=self.colors[j])
                        except Exception as e:
                            print(e)
        img_tk = ImageTk.PhotoImage(image=img, master=self.window)
        self.lab.configure(image=img_tk)
        self.window.update()

    def destroy(self):
        """
        Destroys the opencv image window
        """
        self.window.destroy()
