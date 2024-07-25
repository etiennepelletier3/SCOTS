import numpy as np
import matplotlib.pyplot as plt
from ..utils import multidimensional_plot


class Optical_element:
    """Class encapsulating the method for a mirror optical_element.
    Apodization returns the shape of the Optical_element, round by default"""

    def __init__(self, thickness=0, diameter=np.inf):
        self.__diameter = diameter
        self.__thickness = thickness

    @property
    def diameter(self):
        return self.__diameter

    @property
    def thickness(self):
        return self.__thickness

    def get_apodization(self, radius, angle, wavelength):
        return np.choose(radius <= self.diameter / 2, (0, 1))

    def sag_equation(self, radius, angle, wavelengths):
        """Sag of the element. Takes the coordinates for each wavelengths
        and compute the map for the phase shift"""
        return radius * 0 + 1

    def get_induced_phase_shift(self, coordinates, wavelengths):
        """Method giving the phase shift induced by a transmissive element.
        Here using polar geometry. Wants a coordinates input ordered like:
        [wavelength order,radius/angle,x,y]"""
        radius, angle = coordinates[:, 0, :, :], coordinates[:, 1, :, :]
        dxy = self.sag_equation(radius, angle)
        if np.any(dxy < 0):
            raise (ValueError("Negative in sag"))
        d0 = np.max(np.abs(dxy))
        k0 = 2 * np.pi / wavelengths.reshape(np.size(wavelengths), 1, 1)
        n = self.get_refraction_index(wavelengths.reshape(np.size(wavelengths), 1, 1))
        h0 = np.exp(-1j * k0 * d0)
        hxy = np.exp(-1j * (n - 1) * k0 * dxy)
        return self.get_apodization(radius, angle, wavelengths) * hxy * h0

    def get_refraction_index(self, wavelength):
        """Compute the dispersion coefficien for a given wavelength"""
        return 1

    def get_phase(self, coordinates, wavelengths):
        radius, angle = coordinates[:, 0, :, :], coordinates[:, 1, :, :]
        return (
            self.get_apodization(radius, angle, wavelengths)
            * self.sag_equation(radius, angle)
            * (
                self.get_refraction_index(
                    wavelengths.reshape(np.size(wavelengths), 1, 1)
                )
                - 1
            )
            / (wavelengths.reshape(np.size(wavelengths), 1, 1))
        )

    def get_sag(self, coordinates, wavelengths):
        radius, angle = coordinates[:, 0, :, :], coordinates[:, 1, :, :]
        return self.get_apodization(radius, angle, wavelengths) * self.sag_equation(
            radius, angle
        )

    def display_sag(self, coordinates, colorbar=True):
        """Need to send the coordinates to sag_equation"""
        radius, angle = coordinates[:, 0, :, :], coordinates[:, 1, :, :]
        base_coordinates = np.swapaxes(
            [radius * np.cos(angle), radius * np.sin(angle)], 0, 1
        )
        if coordinates.shape[0] > 1:
            plt.suptitle("Apodization for each wavelengths", y=0.85, fontsize=16)
            multidimensional_plot(
                self.sag_equation(radius, angle),
                coordinates=base_coordinates,
                cmap=plt.cm.gray,
                colorbar=colorbar,
            )
        else:
            # plt.title("Apodization of the pupil plane")
            plt.imshow(
                self.sag_equation(radius, angle)[0],
                # cmap=plt.cm.gray,
                extent=[
                    np.min(base_coordinates),
                    np.max(base_coordinates),
                    np.min(base_coordinates),
                    np.max(base_coordinates),
                ],
            )
            plt.xlabel("Axe x (m)")
            plt.ylabel("Axe y (m)")
            plt.grid(False)
            # plt.colorbar()
            # plt.show()

    def __add__(self, other):
        return Combined_optical_element(self, other)


class Combined_optical_element(Optical_element):
    def __init__(self, optical_element_1, optical_element_2):
        self.optical_element_1 = optical_element_1
        self.optical_element_2 = optical_element_2

    def get_apodization(self, *args):
        return self.optical_element_1.get_apodization(
            *args
        ) * self.optical_element_2.get_apodization(*args)

    def sag_equation(self, *args):
        """Sag of the element. Takes the coordinates for each wavelengths
        and compute the map for the phase shift"""
        return self.optical_element_1.sag_equation(
            *args
        ) + self.optical_element_2.sag_equation(*args)

    def get_induced_phase_shift(self, *args):
        """Method giving the phase shift induced by a transmissive element.
        Here using polar geometry. Wants a coordinates input ordered like:
        [wavelength order,radius/angle,x,y]"""
        return self.optical_element_1.get_induced_phase_shift(
            *args
        ) * self.optical_element_2.get_induced_phase_shift(*args)

    @property
    def diameter(self):
        return np.min(
            (self.optical_element_1.diameter, self.optical_element_2.diameter)
        )

    def get_refraction_index(self, wavelenegth):
        return np.nan

    def __add__(self, other):
        return Combined_optical_element(
            Combined_optical_element(self.optical_element_1, self.optical_element_2),
            other,
        )
