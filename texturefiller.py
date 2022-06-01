
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

class TextureFiller:
    def __init__(self, img, valid, w=11, only_sample_from = None):
        """
        Arguments:
            img: (HxWxC) np array of the image to fill in.
                Must already be the shape of the desired output image.
                The original texture must be present somewhere in this image.
            valid: (HxW) np array indicating which pixels are part of the original texture.
                1 means this pixel is part of the original texture
                0 means this pixel is not part of the original texture, and should be filled.
            w: window width for texture filling
                A smaller window makes the resulting texture more random-looking.
                A larger window makes the resulting texture look more structured.
                w must be odd.
        """

        # Window width must be odd
        assert w % 2 == 1
        self.w = w
        self.w2 = w // 2

        assert img.shape[:2] == valid.shape[:2]

        # Pad the image with w2 pixels on all sides to make calculations easier later
        self.full  = np.zeros(shape=(img.shape[0]+w-1,   img.shape[1]+w-1,   img.shape[2]))
        self.full[self.w2:self.w2+img.shape[0], self.w2:self.w2+img.shape[1]] = img
        self.img_dtype = img.dtype

        # Pad "valid" mask as well
        self.valid = np.zeros(shape=(valid.shape[0]+w-1, valid.shape[1]+w-1, valid.shape[2]), dtype=int)
        self.valid[self.w2:self.w2+valid.shape[0], self.w2:self.w2+valid.shape[1]] = valid

        # Make sure that invalid pixels start as 0
        self.full[np.squeeze(self.valid) == 0] = 0

        # Indicates which pixels still need to be filled in
        self.need_to_fill = np.zeros(shape=self.valid.shape[:2], dtype=int)
        self.need_to_fill[self.w2:-self.w2, self.w2:-self.w2] = 1
        self.need_to_fill[np.squeeze(self.valid) == 1] = 0

        # Compute gaussian kernel only once
        self.gkernel = np.expand_dims(gkern(kernlen=w, std=w//3), -1)

        # Compute counting kernel only once
        self.counting_kernel = np.ones(shape=(self.w, self.w, 1))

        # Compute the valid count only once and then adjust it on the fly
        valid_count = signal.fftconvolve(self.valid, self.counting_kernel, mode='same')
        valid_count = np.around(valid_count).astype(int)
        self.valid_count = np.squeeze(valid_count)

        self.sample_from_synthesized = only_sample_from is None
        self.only_sample_from = None if only_sample_from is None else only_sample_from.astype(float)

        valid_count = signal.fftconvolve(self.valid, self.counting_kernel, mode='same')
        valid_count = np.around(valid_count).astype(int)
        self.valid_count = np.squeeze(valid_count)

        if not self.sample_from_synthesized:
            self.originally_valid = np.expand_dims(np.ones(shape=self.only_sample_from.shape[:2]), axis=-1)
            self.originally_valid = signal.fftconvolve(self.originally_valid, self.counting_kernel, mode='same')
            self.originally_valid = np.squeeze(np.around(self.originally_valid).astype(int))


    def fill_texture(self):
        """
        Call this method to fill in the texture.
        Returns: (HxWxC) np array which is the original image with all the nonvalid pixels filled in.
        """
        def fill_single_pixel():
            i, j = self._get_next_pixel()
            self._fill_pixel(i, j)

        try:
            from tqdm import tqdm
            for _ in tqdm(range(np.sum(self.need_to_fill))):
                fill_single_pixel()
        except ModuleNotFoundError:
            while np.any(self.need_to_fill):
                print("Remaining pixels to fill: ", np.sum(self.need_to_fill))
                fill_single_pixel()

        return self.full[self.w2:-self.w2, self.w2:-self.w2].astype(self.img_dtype)

    def _fill_pixel(self, i, j):
        """ Fill a single pixel with a synthesized texture. i, j are in the padded coordinate system. """

        # First compute 'scores' for all wxw patches in the image
        # The score of a patch is the sum of squared difference with the wxw patch around (i, j),
        # additionally weighed with a gaussian kernel G.
        # Compute as such:
        # Sum[G(A-B)^2] = Sum[G(A^2)] - 2*Sum[GAB] + Sum[G(B^2)]
        # The three components of the above sums can be computed with correlations.
        if self.sample_from_synthesized:
            pic_to_sample_from = self.full
        else:
            pic_to_sample_from = self.only_sample_from

        # wxw patch around (i, j)
        patch_to_match = self.full[i-self.w2:i+self.w2+1, j-self.w2:j+self.w2+1]

        # Compute -2*Sum[GAB]
        patch_match_kernel = self.gkernel * patch_to_match[::-1, ::-1] # important so we're doing a cross-correlation
        scores = -2*signal.fftconvolve(pic_to_sample_from, patch_match_kernel, mode='same', axes=(0, 1))
        scores = np.sum(scores, axis=-1)

        # Compute Sum[G(A^2)]
        patch_normalization_kernel = self.gkernel * self.valid[i-self.w2:i+self.w2+1, j-self.w2:j+self.w2+1]
        sq = pic_to_sample_from * pic_to_sample_from
        sq = signal.fftconvolve(sq, patch_normalization_kernel[::-1, ::-1], mode='same', axes=(0, 1))
        scores += np.sum(sq, axis=-1)

        # Compute Sum[G(B^2)]
        scores += np.sum(patch_to_match * patch_to_match * patch_normalization_kernel)

        WORST_SCORE = 0

        # Don't sample from pixels which aren't valid
        if self.sample_from_synthesized:
            scores[np.squeeze(self.valid) == 0] = WORST_SCORE

        # Don't sample from pixels that have invalid neighbors
        if self.sample_from_synthesized:
            scores[self.valid_count != self.w*self.w] = WORST_SCORE
        else:
            scores[self.originally_valid != self.w*self.w] = WORST_SCORE

        # Turn smaller score is better into larger score is better
        scores[scores == 0] = np.inf
        scores = 1 / scores

        # Sample from pixels within a factor of 1.1 of the best score
        best_score = np.max(scores)
        scores[scores < (1/1.1)*best_score] = 0
        full_flat = np.reshape(pic_to_sample_from, (-1, 3))
        scores_flat = scores.flatten() / np.sum(scores)
        chosen_pixel = full_flat[np.random.choice(full_flat.shape[0], p=scores_flat)]

        self.full[i, j] = chosen_pixel
        self.valid[i, j] = 1
        self.valid_count[i-self.w2:i+self.w2+1, j-self.w2:j+self.w2+1] += 1
        self.need_to_fill[i, j] = 0

    def _get_next_pixel(self):
        # Choose the coordinate with the most valid neighbors
        valid_count = self.valid_count.copy()
        valid_count[self.need_to_fill == 0] = 0
        maxij = np.where(valid_count == np.max(valid_count))
        i = maxij[0][0]
        j = maxij[1][0]
        return i, j
