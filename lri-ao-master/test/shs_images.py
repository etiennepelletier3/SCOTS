import unittest

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path.joinpath(Path(__file__).absolute().parents[1], "src")))

from lri_ao.simulations import AO_simulation_coherent_shs
from lri_ao.optics.bank import Dummy_element


class TestSHSimages(unittest.TestCase):
    config = {}
    config["obscuration_percentage"] = 0.32
    config["wavelengths"] = 632.8e-9
    mag = 1.75
    fnumb = 12
    config["optical_element_list"] = (Dummy_element(), Dummy_element())

    # # CAMÉRA
    config["side_length"] = 10e-3
    config["aperture_diameter"] = config["side_length"] / 2
    config["fnumb_ulentilles"] = 10

    radius_micro = 10e-3
    config["shs_focal_length"] = radius_micro * 3 / (1.4585 - 1)
    config["focal_length_list"] = (1, 1)
    config["optical_element_list"] = (Dummy_element(), Dummy_element())
    config["nb_cote_ulentilles"] = 16 * (
        config["side_length"] // config["aperture_diameter"]
    )
    config["micro_lens_res"] = 5
    config["photon_budget"] = 1e10

    # # CAMÉRA
    config["sampling_points"] = (
        config["micro_lens_res"] * config["nb_cote_ulentilles"] + 1
    )
    config["spatial_resolution"] = config["sampling_points"]

    mode_max = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_simulation = AO_simulation_coherent_shs(**self.config)

    def test1(self):
        self.test_simulation.pupil_plane.phase_change_noll(2, 0)
        self.test_simulation.update()
        x1 = self.test_simulation.fourier_planes[-1].get_irradiance()[0]
        import matplotlib.pyplot as plt

        plt.imshow(np.log(x1))
        plt.show()
        # for x in range(1):
        #     self.test_simulation.pupil_plane.phase_change_noll(x+2, 2e-7)
        #     self.test_simulation.update()
        #     x2 = self.test_simulation.fourier_planes[-1].get_irradiance()[0]
        #     # x = self.test_simulation.pupil_plane.apodization[0]
        #     # self.test_simulation.pupil_plane.apodization[
        #     #     :,
        #     #     : self.test_simulation.pupil_plane.get_coordinates().shape[-1] // 2,
        #     #     : self.test_simulation.pupil_plane.get_coordinates().shape[-1] // 2,
        #     # ] = 0
        #     # self.test_simulation.update()
        #     # x = self.test_simulation.fourier_planes[-1].get_irradiance()[0]
        #     # x = self.test_simulation.pupil_plane.apodization[0]
        #     import matplotlib.pyplot as plt

        #     plt.imshow(x2-x1)
        #     # plt.imshow(x)
        #     plt.show()


if __name__ == "__main__":
    unittest.main()
