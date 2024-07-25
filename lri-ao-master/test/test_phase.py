import numpy as np
from PIL import Image
from pathlib import Path
import unittest

from lri_ao.reconstructors import SHS_AUTO_INIT
from lri_ao.utils import noll_gradZernike, noll_zernike


class TestPhase(unittest.TestCase):
    def test_zernike(self):
        with Image.open((Path(__file__).parents[0]) / "../media/OL-Test0.tiff") as im:
            a = np.asarray(im)
        recon = SHS_AUTO_INIT(a, microlenses_focal=10, thresh=0.15)
        recon.mask = recon.r <= 1
        for x in range(100):
            weights = np.random.random(np.arange(2, 29).shape) - 0.5
            weights /= 2 ** np.arange(2, 29)
            zern = np.array(
                noll_zernike(
                    np.arange(2, 20),
                    recon.r,
                    recon.ang,
                    weight=weights,
                    circle=True,
                )
            )
            zern_g = np.array(
                noll_gradZernike(
                    np.arange(2, 20),
                    recon.r,
                    recon.ang,
                    weight=weights,
                    circle=True,
                )
            )

            zern[~recon.mask] = np.nan
            recon_phase = recon.reconstruction_zonal_cog(zern_g)
            recon_phase = recon_phase / np.nanmean(abs(recon_phase))
            goal = zern / np.nanmean(abs(zern))
            error = abs((goal + 20) - (recon_phase + 20)) / (goal + 20)
            self.assertLess(np.nanmax(error), 25 / 100)
            self.assertLess(np.nanmean(error), 5 / 100)

    def test_image(self):
        with Image.open((Path(__file__).parents[0]) / "../media/OL-Test0.tiff") as im:
            a = np.asarray(im)

        recon = SHS_AUTO_INIT(a, microlenses_focal=10, thresh=0.15)
        recon.mask = recon.r <= 1
        recon.reconstruction_zonal(a)
        recon.reconstruction_modal(a)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(4, 8))
        zonal = recon.reconstruction_zonal(a)
        zonal = zonal / np.nanmax(abs(zonal))
        plt.imshow(zonal)
        plt.colorbar()
        plt.title("OL-Test0 phase, normalised")
        modal_list = recon.reconstruction_modal(a)
        modal_list /= np.nanmax(abs(modal_list))
        plt.xlabel("\n".join([f"z{2+x}:{y:.3E}" for x, y in enumerate(modal_list)]))
        plt.tight_layout()
        plt.clf()
        # plt.show()

    def test_image_modal(self):
        with Image.open((Path(__file__).parents[0]) / "../media/OL-Test0.tiff") as im:
            a = np.asarray(im)

        recon = SHS_AUTO_INIT(a, microlenses_focal=10, thresh=0.15)
        recon.mask = recon.r <= 1
        for x in range(100):
            weights = np.random.random(np.arange(2, 20).shape) - 0.5
            weights *= 10
            weights *= np.exp(-np.arange(2, 20) / 4)

            zern = np.array(
                noll_zernike(
                    np.arange(2, 20),
                    recon.r,
                    recon.ang,
                    weight=weights,
                    circle=True,
                )
            )
            zern_g = np.array(
                noll_gradZernike(
                    np.arange(2, 20),
                    recon.r,
                    recon.ang,
                    weight=weights,
                    circle=True,
                )
            )
            recon_phase = recon.reconstruction_modal_cog(zern_g)

            zern_2 = np.array(
                noll_zernike(
                    np.arange(2, 20),
                    recon.r,
                    recon.ang,
                    weight=recon_phase,
                    circle=True,
                )
            )

            zern[~recon.mask] = np.nan
            zern_2[~recon.mask] = np.nan
            recon_phase = zern_2
            recon_phase = recon_phase / np.nanmean(abs(recon_phase))
            goal = zern / np.nanmean(abs(zern))
            error = abs((goal + 20) - (recon_phase + 20)) / (goal + 20)
            self.assertLess(np.nanmax(error), 25 / 100)
            self.assertLess(np.nanmean(error), 5 / 100)


if __name__ == "__main__":
    # x = TestAutoImage()
    # x.test_auto_init()
    unittest.main()
