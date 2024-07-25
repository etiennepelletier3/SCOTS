import unittest
import numpy as np
from PIL import Image
from pathlib import Path

from lri_ao.reconstructors import SHS_AUTO_INIT


class TestAutoImage(unittest.TestCase):
    def test_auto_init(self):
        try:
            with Image.open(
                (Path(__file__).parents[0]) / "../media/ref_zygo.tiff"
            ) as im:
                a = np.asarray(im)
            shs_object = SHS_AUTO_INIT(
                a, microlenses_focal=10, thresh=0.35, reconstruction_radius=5
            )
            shs_object.cog_measurement(a)[1]
            shs_object.cog_measurement(a)[0]
        except Exception as e:
            self.assertTrue(False, e)

    def test_auto_init2(self):
        try:
            with Image.open(
                (Path(__file__).parents[0]) / "../media/OL-Test0.tiff"
            ) as im:
                a = np.asarray(im)
            shs_object = SHS_AUTO_INIT(a, microlenses_focal=10, thresh=0.5)
            shs_object.cog_measurement(a)[1]
            shs_object.cog_measurement(a)[0]
            print(shs_object.reconstruction(a))
        except Exception as e:
            self.assertTrue(False, e)


if __name__ == "__main__":
    # x = TestAutoImage()
    # x.test_auto_init()
    unittest.main()
