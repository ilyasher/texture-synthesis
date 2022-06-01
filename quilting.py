
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import math

class QuiltTextureFiller:
    def __init__(self, img, out_h, out_w, w=51):
        """
        Arguments:
            img: (HxWxC) np array of texture sample.
            out_h, out_w: Desired dimensions of output image
            w: window width for texture filling
                A smaller window makes the resulting texture more random-looking.
                A larger window makes the resulting texture look more structured.
                w must be odd.
        """
        assert w % 2 == 1
        self.w = w
        self.w2 = w // 2
        self.w6 = w // 6

        self.real_out_h = out_h
        self.real_out_w = out_w

        self.texture = img.copy().astype(float)

        self.out_h = self.w6 + math.ceil((self.real_out_h - self.w6) / (self.w - self.w6)) * (self.w - self.w6)
        self.out_w = self.w6 + math.ceil((self.real_out_w - self.w6) / (self.w - self.w6)) * (self.w - self.w6)
        self.output_texture = np.zeros(shape=(self.out_h, self.out_w, 3))
        self.img_dtype = img.dtype

        self.filled_mask = np.zeros(shape=(self.output_texture.shape[:2]), dtype=int)

        # Compute counting kernel only once
        self.counting_kernel = np.ones(shape=(self.w, self.w, 1))

    def fill_texture(self, blur=False):
        """
        Call this method to fill in the texture.
        Returns: (out_h x out_w x C) np array fo the synthesized texture
        """

        # make coordinates to fill
        coords = list()
        for i in range(self.w2, self.out_h, self.w-self.w6):
            for j in range(self.w2, self.out_w, self.w-self.w6):
                coords.append((i, j))

        # Fill in first patch randomly
        random_i = np.random.randint(0, self.texture.shape[0] - self.w)
        random_j = np.random.randint(0, self.texture.shape[1] - self.w)
        self.output_texture[0:self.w, 0:self.w] = self.texture[random_i:random_i+self.w, random_j:random_j+self.w]
        self.filled_mask[0:self.w, 0:self.w] = 1
        coords.pop(0)

        try:
            from tqdm import tqdm
            for (i, j) in tqdm(coords):
                self._fill_patch_at(i, j, blur=blur)
        except ModuleNotFoundError:
            for num, (i, j) in enumerate(coords):
                print(f"Filling patch {num+1}/{len(coords)}")
                self._fill_patch_at(i, j, blur=blur)

        return self.output_texture.astype(self.img_dtype)[:self.real_out_h, :self.real_out_w]

    def _fill_patch_at(self, i, j, blur=False):

        # wxw patch around (i, j)
        patch_to_match = self.output_texture[i-self.w2:i+self.w2+1, j-self.w2:j+self.w2+1]
        border_mask = np.expand_dims(self.filled_mask[i-self.w2:i+self.w2+1, j-self.w2:j+self.w2+1], axis=-1)

        scores = self.get_patch_scores(self.texture, patch_to_match, border_mask)

        # Incomplete: texture transfer
        # correspondence_scores = self.get_patch_scores(np.mean(self.texture, keepdims=True),
        #                                               np.mean(patch_to_match, keepdims=True),
        #                                               border_mask)

        # scores = alpha * scores + (1 - alpha) * correspondence_scores

        WORST_SCORE = 0

        # Don't sample on the edge
        scores[:self.w2, :] = WORST_SCORE
        scores[-self.w2:, :] = WORST_SCORE
        scores[:, :self.w2] = WORST_SCORE
        scores[:, -self.w2:] = WORST_SCORE

        # Turn smaller score is better into larger score is better
        scores[scores == 0] = np.inf
        scores = 1 / scores

        # Sample from pixels within a factor of 1.1 of the best score
        best_score = np.max(scores)
        scores[scores < (1/1.1)*best_score] = 0
        scores /= np.sum(scores)

        chosen_ij = np.random.choice(self.texture.shape[0] * self.texture.shape[1], p=scores.flatten())
        chosen_i = chosen_ij // self.texture.shape[1]
        chosen_j = chosen_ij % self.texture.shape[1]

        found_patch = self.texture[chosen_i-self.w2:chosen_i+self.w2+1, chosen_j-self.w2:chosen_j+self.w2+1]

        has_horizontal_border = border_mask[0, -1, 0] == 1
        has_vertical_border   = border_mask[-1, 0, 0] == 1
        assert has_horizontal_border or has_vertical_border

        found_patch_mask = np.squeeze(1 - border_mask)

        square_dists = np.sqrt(np.sum((patch_to_match - found_patch) ** 2, axis=-1))

        # Dynamic programming helper function that finds the min cut down the "dists" array
        # "dists" is modified in-place
        def find_mincut(dists):
            dists_choice = np.zeros(shape=(3, self.w6))
            for i in range(found_patch.shape[1]):
                dists_choice[0] = dists[i-1, :-2]
                dists_choice[1] = dists[i-1, 1:-1]
                dists_choice[2] = dists[i-1, 2:]
                dists[i, 1:-1] = dists[i, 1:-1] + np.min(dists_choice, axis=0)

            # Now backtrack to find best path
            j = np.argmin(dists[-1, :])
            dists[-1, :j] = 0
            dists[-1, j:] = 1
            for i in range(self.w-2, -1, -1):
                best_candidate = min((dists[i, j-1], dists[i, j], dists[i, j+1]))
                if dists[i, j-1] == best_candidate:
                    j -= 1
                elif dists[i, j+1] == best_candidate:
                    j += 1
                dists[i, :j] = 0
                dists[i, j:] = 1

        # Find best stitching along the vertical border
        if has_vertical_border:
            dists = np.ones(shape=(self.w, self.w6+2)) * np.inf
            dists[:, 1:-1] = square_dists[:, :self.w6]
            find_mincut(dists)
            found_patch_mask[:, :self.w6] = np.logical_or(dists[:, 1:-1], found_patch_mask[:, :self.w6])

        # Same for the horizontal border
        if has_horizontal_border:
            dists = np.ones(shape=(self.w, self.w6+2)) * np.inf
            dists[:, 1:-1] = np.transpose(square_dists)[:, :self.w6]
            find_mincut(dists)
            found_patch_mask[:self.w6, :] = np.logical_or(np.transpose(dists[:, 1:-1]), found_patch_mask[:self.w6, :])

        # Optional: Blur found_patch_mask
        if blur:
            found_patch_mask = np.expand_dims(gaussian_filter(found_patch_mask.astype(float), sigma=self.w6/4, mode='nearest'), axis=-1)
            found_patch_mask[border_mask == 0] = 1
        else:
            found_patch_mask = np.expand_dims(found_patch_mask, axis=-1)

        # found_patch_mask = np.expand_dims(found_patch_mask, -1)
        patch_to_match[:, :] = found_patch * found_patch_mask + patch_to_match * (1 - found_patch_mask)

        self.filled_mask[i-self.w2:i+self.w2+1, j-self.w2:j+self.w2+1] = 1

    def get_patch_scores(self, sample_from, patch_to_match, valid_mask):
        # First compute 'scores' for all wxw patches in the image
        # The score of a patch is the sum of squared difference with the wxw patch around (i, j),
        # additionally weighed with a gaussian kernel G.
        # Compute as such:
        # Sum[G(A-B)^2] = Sum[G(A^2)] - 2*Sum[GAB] + Sum[G(B^2)]
        # The three components of the above sums can be computed with correlations.

        # Compute -2*Sum[GAB]
        patch_match_kernel = patch_to_match[::-1, ::-1] # important so we're doing a cross-correlation
        scores = -2*signal.fftconvolve(sample_from, patch_match_kernel, mode='same', axes=(0, 1))
        scores = np.sum(scores, axis=-1)

        # Compute Sum[G(A^2)]
        sq = sample_from * sample_from
        sq = signal.fftconvolve(sq, valid_mask[::-1, ::-1], mode='same', axes=(0, 1))
        scores += np.sum(sq, axis=-1)

        # Compute Sum[G(B^2)]
        scores += np.sum(patch_to_match * patch_to_match * valid_mask)

        return scores


