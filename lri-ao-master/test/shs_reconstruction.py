import unittest
import numpy as np

from lri_ao.simulations import AO_simulation_coherent_shs
from lri_ao.optics.bank import Dummy_element
from lri_ao.reconstructors import SHS_reconstructor


class TestSHSreconstruction(unittest.TestCase):
    config_shs = {}
    config_shs["obscuration_percentage"] = 0
    config_shs["wavelengths"] = 632.8e-9
    config_shs["side_length"] = 10e-3
    config_shs["aperture_diameter"] = config_shs["side_length"] / 2
    config_shs["fnumb_ulentilles"] = 10
    radius_micro = 7e-3
    config_shs["shs_focal_length"] = radius_micro * 3 / (1.4585 - 1)
    config_shs["focal_length_list"] = (1, 1)
    config_shs["optical_element_list"] = (Dummy_element(), Dummy_element())
    config_shs["nb_cote_ulentilles"] = 21
    config_shs["micro_lens_res"] = 10
    config_shs["photon_budget"] = 1e10

    # # CAMÉRA
    config_shs["sampling_points"] = (
        config_shs["micro_lens_res"] * config_shs["nb_cote_ulentilles"] + 1
    )
    config_shs["spatial_resolution"] = config_shs["sampling_points"]

    mode_max = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_simulation = AO_simulation_coherent_shs(**self.config_shs)
        self.test_reconstruction = SHS_reconstructor(
            self.config_shs["nb_cote_ulentilles"],
            self.config_shs["sampling_points"],
            microlenses_focal=self.config_shs["shs_focal_length"],
            microlenses_pitch=500e-6,
            reconstruction_radius=self.config_shs["aperture_diameter"]
            / self.config_shs["side_length"],
            maxmode=self.mode_max,
        )

    def no_test_cog(self):
        for x in range(10):
            nb_ulentilles = np.random.randint(10, 50, 1)[0]
            res_micro = np.random.randint(10, 25, 1)[0]
            image_resolution = nb_ulentilles * res_micro
            test_reconstruction_local = SHS_reconstructor(
                nb_ulentilles,
                image_resolution,
                reconstruction_radius=self.config_shs["aperture_diameter"],
            )

            cog_test = np.random.random(
                (
                    2,
                    nb_ulentilles,
                    nb_ulentilles,
                )
            )
            cog_test *= res_micro / 2 - 2
            cog_test += res_micro / 2
            cog_test[0, :, 3] = res_micro / 2
            cog_test = cog_test.astype("int")
            

            cog_input = np.copy(cog_test)
            cog_input += np.array(
                np.mgrid[
                    slice(0, nb_ulentilles, 1),
                    slice(0, nb_ulentilles, 1),
                ]
                * res_micro
            )

            image_test = np.zeros((image_resolution, image_resolution))
            image_test[cog_input[0], cog_input[1]] = 1


            cog_result = test_reconstruction_local.cog_measurement(image_test)

            self.assertTrue(
                np.all(
                    np.isclose(
                        cog_result[0],
                        cog_test[0] - res_micro // 2,
                    )
                )
            )

    def no_test_cog_single(self):
        for x in range(10):
            nb_ulentilles = 1
            res_micro = 250
            image_resolution = nb_ulentilles * res_micro
            test_reconstruction_local = SHS_reconstructor(
                nb_ulentilles,
                image_resolution,
                reconstruction_radius=self.config_shs["aperture_diameter"],
            )

            cog_test = np.random.random(
                (
                    2,
                    nb_ulentilles,
                    nb_ulentilles,
                )
            )
            cog_test *= res_micro / 2 - 2
            cog_test += res_micro / 2
            # breakpoint()
            cog_test[0, :, -1] = res_micro / 2
            cog_test = cog_test.astype("int")

            cog_input = np.copy(cog_test)
            cog_input += np.array(
                np.mgrid[
                    slice(0, nb_ulentilles, 1),
                    slice(0, nb_ulentilles, 1),
                ]
                * res_micro
            )

            image_test = np.zeros((image_resolution, image_resolution))
            image_test[cog_input[0], cog_input[1]] = 1

            cog_result = test_reconstruction_local.cog_measurement(image_test)
            self.assertTrue(
                np.all(
                    np.isclose(
                        cog_result[0],
                        cog_test[0] - res_micro // 2,
                    )
                )
            )

    def test_recon_shs(self):
        # self.test_simulation.pupil_plane.phase_change_noll((5), 3e-6)
        self.test_simulation.update()
        imgbase = np.sum(
            self.test_simulation.detector_plane.get_zoomed_irradiance(), axis=0
        )
        self.test_reconstruction.add_flat(imgbase)
        recon_array = np.zeros((self.mode_max - 1, self.mode_max - 1))

        for index, mode in enumerate(recon_array):
            if index < 10:
                self.test_simulation.pupil_plane.phase_change_noll((index + 2), 1e-6)
                self.test_simulation.update()
                img = np.sum(
                    self.test_simulation.detector_plane.get_zoomed_irradiance(), axis=0
                )
                import matplotlib.pyplot as plt

                # self.test_simulation.fourier_planes[1].display_irradiance()

                recon_array[index] = self.test_reconstruction.reconstruction(
                    img, mode_max=self.mode_max
                )

                gy, gx = self.test_reconstruction.cog_measurement(img)
                ty, tx = self.test_reconstruction.theoritical_slope_mode(index + 2)
                fig, axs = plt.subplots(2, 2)
                axs[0, 0].imshow(gx)
                axs[0, 1].imshow(-1 * tx)
                axs[1, 0].imshow(gy)
                axs[1, 1].imshow(-1 * ty)
                plt.show()
                # plt.clf()

        plt.imshow(recon_array)
        plt.show()


if __name__ == "__main__":
    unittest.main()
