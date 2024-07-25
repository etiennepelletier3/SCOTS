import numpy as np
from .utils import zoom


class Perfect_Camera:
    def read(self, irradiance, wavelengths):
        photon_read = np.sum(np.random.poisson(photon_read), axis=0)
        photon_read = np.sum(irradiance, axis=0)
        if np.sum(photon_read) != 0:
            photon_read = photon_read * np.sum(photon_read) / np.sum(photon_read)
        else:
            photon_read = photon_read * np.sum(photon_read)
        # Adding Sensor specific noise
        return photon_read


class CCD_camera:
    """Class encapsulating the different methods used to simulate the image
    from a CCD taking an Electric_field from an input. The output from the read
    function returns the
    """

    def __init__(
        self,
        gain=5000,
        adc_resolution=2 ** 16,
        full_well_capacity=800 * 1000,
        read_noise=0.1,
        integration_time=1 / 200,
        spatial_resolution=128,
        dark_current=0.0006,
        clock_induced_charges=0.005,
    ):
        if gain < 1:
            raise AssertionError("Gain cannot be lower than 0")
        self.__adc_resolution = adc_resolution
        self.__gain = gain
        self.__full_well_capacity = full_well_capacity
        self.__integration_time = integration_time
        self.__spatial_resolution = spatial_resolution
        self.__dark_current = dark_current
        self.__clock_induced_charges = clock_induced_charges
        self.__read_noise = read_noise

    def get_quantum_efficiency(self, wavelengths):
        """Method that gives the quantum efficiency in function of the
        wavelength."""
        return 0 * np.array(wavelengths).reshape(np.size(wavelengths), 1, 1) + 1

    def add_detector_noise(self, array):
        return abs(
            array + np.random.normal(0, (self.__read_noise) * np.ones(array.shape))
        )

    def read(self, irradiance, wavelengths):
        """Takes irradiance as photon number per second
        Based on HAMAMATSU Javascript simulations"""
        irr_res = irradiance.shape[-1]
        # How much photon is getting to each pixels
        # Reshape to get the number of pixel of the camera
        photon_read = abs(
            (
                self.__integration_time
                * irradiance
                * self.get_quantum_efficiency(wavelengths).reshape(
                    np.size(wavelengths), 1, 1
                )
            )
        )
        # Adding photon noise and taking out the wavelength dependence
        photon_read = np.sum(np.random.poisson(photon_read), axis=0)
        photon_read = np.sum(photon_read, axis=0)
        # Zoom array to get the same number of pixel as nüvu
        photon_read_zoom = zoom(photon_read, self.__spatial_resolution / irr_res)
        # Scale to keep same flux overall
        if np.sum(photon_read_zoom) != 0:
            photon_read = (
                photon_read_zoom * np.sum(photon_read) / np.sum(photon_read_zoom)
            )
        else:
            photon_read = photon_read_zoom * np.sum(photon_read)
        # Adding Sensor specific noise
        photon_read = (
            photon_read
            + self.__dark_current * self.__integration_time
            + self.__clock_induced_charges
        )
        photon_read = np.clip(np.floor(photon_read), 0, self.__full_well_capacity)
        # compute electron noise and
        # electron_read = np.floor(photon_read*self.__gain/5000)*self.__gain
        electron_read_noise = self.add_detector_noise(photon_read) * self.__gain
        # Make it clip
        read_clip = np.clip(
            np.floor(
                electron_read_noise * self.__adc_resolution / self.__full_well_capacity
            ),
            0,
            self.__adc_resolution,
        )
        # returns in % of pixel saturation. Not exactly as nuvu gives the data
        # back
        return read_clip
