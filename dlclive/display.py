"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import circle

class Display(object):
    '''
    Simple object to display frames with DLC labels.

    Parameters
    -----------
    cmap : string
        string indicating the Matoplotlib colormap to use.
    '''

    def __init__(self, cmap='winter'):

        self.cmap = cmap
        self.colors = None

    def set_colors(self, n_colors):
        '''
        Set the colors for keypoints

        Parameters
        -----------
        n_colors : int
            The number of colors (or number of points)
        '''

        colorclass = plt.cm.ScalarMappable(cmap=self.cmap)
        C = colorclass.to_rgba(np.linspace(0, 1, n_colors))
        self.colors = (C[:,:3]*255).astype(np.uint8)

    def display_frame(self, frame, pose):
        '''
        Display the image with DeepLabCut labels using opencv imshow

        Parameters
        -----------
        frame :class:`numpy.ndarray`
            an image as a numpy array

        pose :class:`numpy.ndarray`
            the pose estimated by DeepLabCut for the image
        '''

        if self.colors is None:
            self.set_colors(pose.shape[0])

        im_size = (frame.shape[1], frame.shape[0])

        for i in range(pose.shape[0]):
            if pose[i,2] > 0.75:
                rr, cc = circle(pose[i,1], pose[i,0], 3, shape=im_size)
                rr[rr > im_size[0]] = im_size[0]
                cc[cc > im_size[1]] = im_size[1]
                try:
                    frame[rr, cc, :] = self.colors[i]
                except:
                    pass

        cv2.imshow('DLC Live', frame)
        cv2.waitKey(1)

    def destroy(self):
        '''
        Destroys the opencv image window
        '''
        
        cv2.destroyAllWindows()
