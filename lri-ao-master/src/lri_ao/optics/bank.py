import numpy as np
from scipy.signal import sawtooth
from .__init__ import Optical_element
from ..utils import noll_zernike

# import matplotlib.pyplot as plt

# from ao_sim_util import *
# from util import *
# from light_util import *


def dispersion_formula_fused_silica(wavelength):
    wavelength = wavelength * 1e6
    return np.sqrt(
        (0.6961663 * wavelength ** 2 / (wavelength ** 2 - 0.0684043 ** 2))
        + (0.4079426 * wavelength ** 2 / (wavelength ** 2 - 0.1162414 ** 2))
        + (0.8974794 * wavelength ** 2 / (wavelength ** 2 - 9.896161 ** 2))
        + 1
    )


def dispersion_formula_nbk7(wavelength):
    wavelength = wavelength * 1e6
    return np.sqrt(
        (1.03961212 * wavelength ** 2 / (wavelength ** 2 - 0.00600069867))
        + (0.231792344 * wavelength ** 2 / (wavelength ** 2 - 0.0200179144))
        + (1.01046945 * wavelength ** 2 / (wavelength ** 2 - 103.560653))
        + 1
    )


class Mirror(Optical_element):
    """Class encapsulating the method for a mirror optical_element."""

    def induced_phase_shift(self, coordinates, wavelengths):
        return super.induced_phase_shift(coordinates, wavelengths) ** 2


class Dummy_element(Optical_element):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_induced_phase_shift(self, coordinates, wavelengths):
        return 1

    def get_apodization(self, radius, angle, wavelengths):
        return 1


class Axicon(Optical_element):
    def __init__(self, angle, decenter=(0, 0), **kwargs):
        """decenter argument is (dx, dy)"""
        super().__init__(**kwargs)
        self.__angle = angle
        self.__decenter = decenter

    @property
    def decenter(self):
        return self.__decenter

    @property
    def angle(self):
        return self.__angle

    def sag_equation(self, radius, angle):
        x_axis = np.cos(angle) * radius
        y_axis = np.sin(angle) * radius
        x_axis += self.decenter[0]
        y_axis += self.decenter[1]
        radius = np.sqrt(x_axis ** 2 + y_axis ** 2)
        return np.sin(self.angle) * radius + self.thickness

    def get_refraction_index(self, wavelength):
        """Compute the dispersion coefficient for a given wavelength"""
        return dispersion_formula_fused_silica(wavelength)


class Spiral(Optical_element):
    def __init__(self, center_wavelength=632.8e-9, order=1, decenter=(0, 0), **kwargs):
        """decenter argument is (dx, dy)"""
        super().__init__(**kwargs)
        self.__center_wavelength = center_wavelength
        self.__decenter = decenter
        self.__order = order

    @property
    def decenter(self):
        return self.__decenter

    @property
    def center_wavelength(self):
        return self.__center_wavelength

    @property
    def order(self):
        return self.__order

    def sag_equation(self, radius, angle):
        x_axis = np.cos(angle) * radius
        y_axis = np.sin(angle) * radius
        x_axis += self.decenter[0]
        y_axis += self.decenter[1]
        angle = np.arctan2(y_axis, x_axis)
        angle = angle - np.min(angle)
        angle = angle / np.max(angle)
        return (
            angle
            * self.center_wavelength
            * self.order
            / ((dispersion_formula_fused_silica(self.center_wavelength) - 1))
        )

    def get_refraction_index(self, wavelength):
        """Compute the dispersion coefficient for a given wavelength"""
        return dispersion_formula_fused_silica(wavelength)


class Pyramid(Optical_element):
    def __init__(self, angle, edge_flatness=0, decenter=(0, 0), **kwargs):
        """decenter argument is (dx, dy)"""
        super().__init__(**kwargs)
        self.__angle = angle
        self.__edge_flatness = edge_flatness
        self.__decenter = decenter

    @property
    def decenter(self):
        return self.__decenter

    @property
    def angle(self):
        return self.__angle

    @property
    def edge_flatness(self):
        return self.__edge_flatness

    def get_apodization(self, radius, angle, wavelengths):
        return np.choose(
            (abs(np.cos(angle)) * radius <= self.diameter / 2)
            * (abs(np.sin(angle)) * radius <= self.diameter / 2),
            (0, 1),
        )

    def sag_equation(self, radius, angle):
        xx = abs(np.cos(angle) * radius + self.decenter[0])
        yy = abs(np.sin(angle) * radius + self.decenter[1])
        x_axis = np.clip(xx, self.edge_flatness, np.inf)
        x_axis -= np.min(x_axis)
        y_axis = np.clip(yy, self.edge_flatness, np.inf)
        y_axis -= np.min(y_axis)
        return ((x_axis + y_axis) * np.sin(self.angle) / np.sqrt(2)) + self.thickness

    def get_refraction_index(self, wavelength):
        """Compute the dispersion coefficient for a given wavelength"""
        return dispersion_formula_nbk7(wavelength)


class Zernike_phase(Mirror):
    def __init__(self, design_wavelength, size, **kwargs):
        super().__init__(**kwargs)
        self.__design_wavelength = design_wavelength
        self.__size = size

    @property
    def size(self):
        return self.__size

    @property
    def design_wavelength(self):
        return self.__design_wavelength

    def sag_equation(self, radius, angle):
        phase = (radius < self.size / 2) * 1
        return phase * np.pi * self.design_wavelength / 2

    def get_refraction_index(self, wavelength):
        """Compute the dispersion coefficient for a given wavelength"""
        return dispersion_formula_nbk7(wavelength)


class Grating(Optical_element):
    def __init__(self, angle, frequency=0, decenter=(0, 0), **kwargs):
        """decenter argument is (dx, dy)"""
        super().__init__(**kwargs)
        self.__angle = angle
        self.__decenter = decenter

    @property
    def decenter(self):
        return self.__decenter

    @property
    def angle(self):
        return self.__angle

    def get_apodization(self, radius, angle, wavelengths):
        return np.choose(
            (abs(np.cos(angle)) * radius <= self.diameter / 2)
            * (abs(np.sin(angle)) * radius <= self.diameter / 2),
            (0, 1),
        )

    def sag_equation(self, radius, angle):
        xx = np.cos(angle) * radius + self.decenter[0]
        yy = np.sin(angle) * radius + self.decenter[1]

        return (
            sawtooth(
                xx
                * 2
                * np.pi
                * ((xx.max() - xx.min()) / (np.tan(70 * np.pi / 180) * 2e-6))
                / np.max(xx)
            )
            * 2e-6
        )

    def get_refraction_index(self, wavelength):
        """Compute the dispersion coefficient for a given wavelength"""
        return dispersion_formula_nbk7(wavelength)


class ThinPlate(Optical_element):
    def __init__(self, thickness, decenter=(0, 0), **kwargs):
        """decenter argument is (dx, dy)"""
        super().__init__(**kwargs)
        self.__thickness = thickness
        self.__decenter = decenter

    @property
    def decenter(self):
        return self.__decenter

    @property
    def thickness(self):
        return self.__thickness

    def sag_equation(self, radius, angle):
        phase = abs(np.ones_like(radius) * self.thickness)
        return phase

    def get_refraction_index(self, wavelength):
        """Compute the dispersion coefficient for a given wavelength"""
        return dispersion_formula_nbk7(wavelength)


class Zernike_phase_plate(Optical_element):
    """Class that emulates the behaviour of a Zernike Phase Plate. Thickness
    is constant thickness of the plate, indices a (n) length vector with
    the zernike indices and weights a (n) length vector with the associated
    weights in waves associated with the indices. Weights must be a list
    of function where f(wavelength) = weight"""

    def __init__(
        self,
        thickness=0,
        indices=np.ones((1,)),
        weights=[lambda x: np.ones(np.shape(x)),],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.__thickness = thickness
        self.__indices = indices
        self.__weights = weights

    @property
    def thickness(self):
        return self.__thickness

    @property
    def indices(self):
        return self.__indices

    @indices.setter
    def indices(self, value):
        self.__indices = value

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, value):
        if len(value) != len(self.indices):
            raise (ValueError("Weights length must be the same as indices"))
        self.__weights = value

    def get_induced_phase_shift(self, coordinates, wavelengths):
        """Method giving the phase shift induced by a transmissive element.
        Here using polar geometry. Wants a coordinates input ordered like:
        [wavelength order,radius/angle,x,y]"""
        radius, angle = coordinates[:, 0, :, :], coordinates[:, 1, :, :]
        dxy = self.sag_equation(radius, angle)
        weights_temp = np.array(
            [weights_ele(wavelengths) for weights_ele in self.weights]
        ).T
        phase_zernike = np.array(
            [
                noll_zernike(
                    self.indices, radius[index], angle[index], weights_temp[index]
                )
                for index in range(len(wavelengths))
            ]
        )
        if np.any(dxy < 0):
            raise (ValueError("Negative in sag"))
        d0 = np.max(np.abs(dxy))
        k0 = 2 * np.pi / wavelengths.reshape(np.size(wavelengths), 1, 1)
        n = self.get_refraction_index(wavelengths.reshape(np.size(wavelengths), 1, 1))
        h0 = np.exp(-1j * k0 * d0)
        hxy = np.exp(-1j * (n - 1) * (k0 * dxy + 2 * np.pi * phase_zernike))
        return self.get_apodization(radius, angle, wavelengths) * hxy * h0

    def sag_equation(self, radius, angle):
        phase = abs(np.ones_like(radius) * self.thickness)
        return phase

    def get_refraction_index(self, wavelength):
        """Compute the dispersion coefficient for a given wavelength"""
        return dispersion_formula_nbk7(wavelength)


if __name__ == "__main__":
    # a = Pupil()
    # b = Axicon(-0.5*np.pi/180)
    a = Zernike_phase_plate()
    a.indices = (2, 3)
    a.weights = (lambda x: 0 if x < 700 else 1, lambda x: 3 if x < 700 else 4)
    # plt.imshow(a.phase)
    # print(dispersion_formula_nbk7(500e-9))
    # plt.imshow(
    #            b.sag_equation(a.get_polar_coordinates()[0, 0, :, :],
    #                                   a.get_polar_coordinates()[0, 1, :, :]) *
    #            b.get_apodization(a.get_polar_coordinates()[0, 0, :, :],
    #                              a.get_polar_coordinates()[0, 1, :, :])
    #            )
    # plt.colorbar()
    # plt.show()
