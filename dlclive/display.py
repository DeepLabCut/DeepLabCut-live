"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""


from tkinter import Tk, Label
import colorcet as cc
from PIL import Image, ImageTk, ImageDraw


class Display(object):
    """
    Simple object to display frames with DLC labels.

    Parameters
    -----------
    cmap : string
        string indicating the Matoplotlib colormap to use.
    pcutoff : float
        likelihood threshold to display points
    """

    def __init__(self, cmap="bmy", radius=3, pcutoff=0.5):
        """ Constructor method
        """

        self.cmap = cmap
        self.colors = None
        self.radius = radius
        self.pcutoff = pcutoff
        self.window = None

    def set_display(self, im_size, bodyparts):
        """ Create tkinter window to display image
        
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
        self.colors = all_colors[:: int(len(all_colors) / bodyparts)]

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

        im_size = (frame.shape[1], frame.shape[0])

        if pose is not None:

            if self.window is None:
                self.set_display(im_size, pose.shape[0])

            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)

            for i in range(pose.shape[0]):
                if pose[i, 2] > self.pcutoff:
                    try:
                        x0 = (
                            pose[i, 0] - self.radius
                            if pose[i, 0] - self.radius > 0
                            else 0
                        )
                        x1 = (
                            pose[i, 0] + self.radius
                            if pose[i, 0] + self.radius < im_size[1]
                            else im_size[1]
                        )
                        y0 = (
                            pose[i, 1] - self.radius
                            if pose[i, 1] - self.radius > 0
                            else 0
                        )
                        y1 = (
                            pose[i, 1] + self.radius
                            if pose[i, 1] + self.radius < im_size[0]
                            else im_size[0]
                        )
                        coords = [x0, y0, x1, y1]
                        draw.ellipse(
                            coords, fill=self.colors[i], outline=self.colors[i]
                        )
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
