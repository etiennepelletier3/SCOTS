#!/usr/bin/env python

import numpy as np

import matplotlib.pyplot as plt
from .utils import Coordinate_system
from .utils import zoom, pad_with, multidimensional_plot
from .utils import noll_zernike, zernike, fourier


class Electric_field(Coordinate_system):
    """Packages the computation used for defining an electric field wavefront.
    Inherit coordinate systemdefinition from Coordinate_system class.

    Gets initialised with neutral phase and intensity.
    Apodization is (in this case) a binary map that takes makes the field
    to 0. Can be initialised with another coordinate system provided
    by Electric_field's 'get_propagated_coordinates'(using init_coordinates
    kwarg).
    If __init__ with init_coordinates, you must have the same number of
    wavelengths as coordinates"""

    def __init__(self, wavelengths=642e-9, photon_budget=1, carrier=1, **kwargs):
        self.__wavelengths = np.atleast_1d(np.array(wavelengths))
        super().__init__(**kwargs)
        # print(self.sampling_points)
        if np.size(self.wavelengths) != self.get_coordinates(self.wavelengths).shape[0]:
            raise (
                ValueError(
                    "Must have same number of wavelengths \
                             and coordinates"
                )
            )

        # Constructing empty phase using the normalised_coordinates
        self.__phase = np.sum(self.get_coordinates(self.wavelengths), axis=1) * 0
        # Constructing intensity of one when summing all wavelengths
        self.__intensity = (self.__phase + 1) / np.sqrt(np.size(self.wavelengths))
        self.__apodization = self.__phase + 1
        self.__photon_budget = np.array(photon_budget)
        self.__carrier = np.array(carrier)

    def get_propagated_coordinates(self, focal_length):
        current_coordinates = self.get_coordinates(self.wavelengths)
        # print(current_coordinates.shape)
        # return current_coordinates
        max_coord = np.max(current_coordinates, axis=(1, 2, 3)).reshape(
            np.size(self.wavelengths), 1, 1, 1
        )
        wavelengths = self.wavelengths.reshape(np.size(self.wavelengths), 1, 1, 1)
        # print((self.sampling_points-1)/(2*self.side_length*self.sampling_points))
        frequencies = (
            current_coordinates * (self.sampling_points - 1) / (4 * max_coord ** 2)
        )
        return frequencies * wavelengths * focal_length

    @property
    def photon_budget(self):
        return self.__photon_budget

    @property
    def carrier(self):
        return self.__carrier

    @property
    def wavelengths(self):
        return self.__wavelengths

    @property
    def phase(self):
        """Property of the phase object"""
        return self.__phase

    @phase.getter
    def phase(self):
        return self.__phase * self.apodization

    @phase.setter
    def phase(self, new_phase):
        """Sets phase with the good shape"""
        new_phase = np.array(new_phase)
        if self.phase.shape == new_phase.shape:
            self.__phase = new_phase
        else:
            try:
                self.__phase = np.tile(
                    new_phase.T, np.size(self.wavelengths)
                ).T.reshape(
                    np.size(self.wavelengths),
                    self.get_coordinates().shape[-2],
                    self.get_coordinates().shape[-1],
                )
            except ValueError:
                raise ValueError(
                    "New phase doesn't have the right dimensions \
                                 ({2} by {0} by {1})".format(
                        self.get_coordinates().shape[-2],
                        self.get_coordinates().shape[-1],
                        np.size(self.wavelengths),
                    )
                )

    @property
    def apodization(self):
        return self.__apodization

    @apodization.setter
    def apodization(self, new_apod):
        apodization = np.array(new_apod)
        if self.apodization.shape == apodization.shape:
            self.__apodization = apodization
        else:
            try:
                self.__apodization = np.tile(
                    apodization.T, np.size(self.wavelengths)
                ).T.reshape(
                    np.size(self.wavelengths),
                    self.get_coordinates().shape[-2],
                    self.get_coordinates().shape[-1],
                )
            except ValueError:
                raise ValueError(
                    "New apodization doesn't have the right \
                                 dimensions ({2} by {0} by {1})".format(
                        self.get_coordinates().shape[-2],
                        self.get_coordinates().shape[-1],
                        np.size(self.wavelengths),
                    )
                )

    @property
    def intensity(self):
        return self.__intensity * self.apodization

    @intensity.setter
    def intensity(self, new_intensity):
        intensity = np.array(new_intensity)
        if self.intensity.shape == intensity.shape:
            self.__intensity = intensity
        else:
            try:
                self.__intensity = np.tile(
                    intensity.T, np.size(self.wavelengths)
                ).T.reshape(
                    np.size(self.wavelengths),
                    self.get_coordinates().shape[-2],
                    self.get_coordinates().shape[-1],
                )
            except ValueError:
                raise ValueError(
                    "New intensity doesn't have the right \
                                 dimensions ({2} by {0} by {1})".format(
                        self.get_coordinates().shape[-2],
                        self.get_coordinates().shape[-1],
                        np.size(self.wavelengths),
                    )
                )

    def change_phase(self, new_phase):
        """Allows to change the phase to a new one"""
        self.phase = new_phase

    def update(self, new_field):
        self.phase = (
            np.angle(new_field)
            * self.wavelengths.reshape(np.size(self.wavelengths), 1, 1)
        ) / (2 * np.pi)
        self.intensity = np.abs(new_field)

    def reset_phase(self):
        """Allows to reset phase to zero"""
        self.phase = np.sum(self.get_coordinates, axis=1) * 0

    def add_phase(self, added_phase):
        added_phase = np.array(added_phase)
        if self.phase.shape == added_phase.shape:
            self.phase = self.phase + added_phase
        else:
            try:
                self.phase = self.phase + np.tile(
                    added_phase.T, np.size(self.wavelengths)
                ).T.reshape(
                    np.size(self.wavelengths),
                    self.get_coordinates().shape[-2],
                    self.get_coordinates().shape[-1],
                )
            except ValueError:
                raise ValueError(
                    "New phase doesn't have the right dimensions \
                                 ({2} by {0} by {1})".format(
                        self.get_coordinates().shape[-2],
                        self.get_coordinates().shape[-1],
                        np.size(self.wavelengths),
                    )
                )

    def get_field(self):
        """Compute and returns the electric field with intensity
        from the intensity and the phase for each wavelength."""
        return self.intensity * np.exp(
            -2j
            * self.phase
            * np.pi
            / self.wavelengths.reshape(np.size(self.wavelengths), 1, 1)
        )

    def get_irradiance(self):
        """Compute the irradiance of each electric field wavelength."""
        return abs(self.get_field()) ** 2

    def get_zoomed_irradiance(self, **kwargs):
        return ((self.get_zoomed_field(**kwargs)) ** 2).astype(float)

    def get_zoomed_field(self, ref=None):
        """Compute the total irradiance when summing all electric fields
        Will shrink the smallest axis when not equivalent in order to
        fit all coordinates system"""
        max_coordinates = np.max(self.get_coordinates(self.wavelengths), axis=(1, 2, 3))
        if ref is not None:
            zoom_values = max_coordinates / (
                max_coordinates[0] * ref / self.wavelengths[0]
            )
        else:
            zoom_values = max_coordinates / np.max(max_coordinates)
        field_temps = self.get_field()
        if np.all(np.isclose(zoom_values, 1)):
            return field_temps
        else:
            zoomed_array = np.ones(field_temps.shape) + (1 + 1j)
            for z in range(np.size(zoom_values)):
                if zoom_values[z] == 1:
                    zoomed_array_temp = field_temps[z]
                else:
                    if self.line_array is True:
                        if self.sampling_points % 2:
                            zoomed_array_temp = zoom(
                                field_temps[z],
                                (
                                    (
                                        np.floor(
                                            zoom_values[z] * self.sampling_points / 2
                                        )
                                        * 2
                                        + 1
                                    )
                                    / self.sampling_points,
                                    1,
                                ),
                                order=0,
                            )
                        else:
                            zoomed_array_temp = zoom(
                                field_temps[z],
                                (
                                    np.floor(zoom_values[z] * self.sampling_points / 2)
                                    * 2
                                    / self.sampling_points,
                                    1,
                                ),
                                order=0,
                            )
                        width = int(
                            np.ceil(
                                (self.sampling_points - zoomed_array_temp.shape[0]) / 2
                            )
                        )
                        zoomed_array_temp = np.pad(
                            zoomed_array_temp,
                            ((width, width), (0, 0)),
                            "constant",
                            constant_values=0,
                        )
                    else:
                        if self.sampling_points % 2:
                            zoomed_array_temp = zoom(
                                field_temps[z],
                                (
                                    (
                                        np.floor(
                                            zoom_values[z] * self.sampling_points / 2
                                        )
                                        * 2
                                        + 1
                                    )
                                    / self.sampling_points,
                                    (
                                        np.floor(
                                            zoom_values[z] * self.sampling_points / 2
                                        )
                                        * 2
                                        + 1
                                    )
                                    / self.sampling_points,
                                ),
                                order=0,
                            )
                        else:
                            zoomed_array_temp = zoom(
                                field_temps[z],
                                (
                                    np.floor(zoom_values[z] * self.sampling_points / 2)
                                    * 2
                                    / self.sampling_points,
                                    np.floor(zoom_values[z] * self.sampling_points / 2)
                                    * 2
                                    / self.sampling_points,
                                ),
                                order=0,
                            )
                        if zoomed_array_temp.shape[-1] < self.sampling_points:
                            zoomed_array_temp = np.pad(
                                zoomed_array_temp,
                                int(
                                    np.ceil(
                                        (
                                            self.sampling_points
                                            - zoomed_array_temp.shape[-1]
                                        )
                                        / 2
                                    )
                                ),
                                pad_with,
                            )
                        elif zoomed_array_temp.shape[-1] > self.sampling_points:
                            half = zoomed_array_temp.shape[-1] // 2
                            quarter = self.sampling_points // 2
                            if self.sampling_points % 2:
                                zoomed_array_temp = zoomed_array_temp[
                                    half - quarter : half + quarter + 1,
                                    half - quarter : half + quarter + 1,
                                ]
                            else:
                                zoomed_array_temp = zoomed_array_temp[
                                    half - quarter : half + quarter,
                                    half - quarter : half + quarter,
                                ]
                        # import matplotlib.pyplot as plt
                        # plt.imshow(abs(zoomed_array_temp)**2)
                        # plt.show()
                    # if zoomed_array.shape[-1] != field_temps.
                    # if zoomed_array_temp.shape[0] > field_temps.shape[1]:
                    #     if self.line_array is True:
                    #         zoomed_array_temp = zoomed_array_temp[1:,:]
                    #     else:
                    #         zoomed_array_temp = (zoomed_array_temp[1:, 1:]+
                    #                              zoomed_array_temp[:-1, :-1]+
                    #                              zoomed_array_temp[1:, :-1]+
                    #                              zoomed_array_temp[:-1, 1:])/2
                zoomed_array[z, :, :] = zoomed_array_temp
            return zoomed_array

    def get_normalised_irradiance(self, **kwargs):
        return abs(self.get_normalised_field(**kwargs)) ** 2

    def get_carrier_normalised_field(self, **kwargs):
        """Normalised with photon budget"""
        photon_budget = self.photon_budget
        if np.all(photon_budget == 1):
            photon_budget = np.ones(np.size(self.wavelengths))
        unnormalised_field = self.get_zoomed_field(**kwargs)
        unnormalised_irradiance = (abs(unnormalised_field))
        normalisation = np.nanmax(unnormalised_irradiance, axis=(1, 2)).reshape(
            np.size(self.wavelengths), 1, 1
        )
        if np.all(normalisation) != 0:
            normalised_field = unnormalised_field / normalisation
        else:
            normalised_field = unnormalised_field
        photon_normalisation = np.reshape(
            photon_budget, (np.size(self.wavelengths), 1, 1)
        )
        return normalised_field * photon_normalisation

    def get_normalised_field(self, **kwargs):
        """Normalised with photon budget"""
        photon_budget = self.photon_budget
        if np.all(photon_budget == 1):
            photon_budget = np.ones(np.size(self.wavelengths))
        unnormalised_field = self.get_zoomed_field(**kwargs)
        unnormalised_irradiance = abs(unnormalised_field) ** 2
        normalisation = np.nansum(unnormalised_irradiance, axis=(1, 2)).reshape(
            np.size(self.wavelengths), 1, 1
        )
        if np.all(normalisation) != 0:
            normalised_field = unnormalised_field / np.sqrt(normalisation)
        else:
            normalised_field = unnormalised_field
        photon_normalisation = np.reshape(
            photon_budget, (np.size(self.wavelengths), 1, 1)
        )
        return normalised_field * np.sqrt(photon_normalisation)

    def display_irradiance(self, colorbar=True):
        if np.size(self.wavelengths) <= 1:
            wavelength_list = [
                self.wavelengths,
            ]
        else:
            wavelength_list = self.wavelengths
        plt.suptitle("Detector irradiance for each wavelengths", y=0.85, fontsize=16)
        multidimensional_plot(
            self.get_irradiance(),
            titles_list=[
                str(wavelength * 1e6) + " µm" for wavelength in wavelength_list
            ],
            coordinates=self.get_coordinates(self.wavelengths),
            cmap=plt.cm.gray,
            colorbar=colorbar,
        )

    def display_total_irradiance(self, colorbar=True):
        photon_budget = self.photon_budget
        if np.all(photon_budget == 1):
            photon_budget = np.ones(np.size(self.wavelengths))
        # plt.title("Detector irradiance")
        base_coordinates = self.get_coordinates(self.wavelengths)
        plt.imshow(
            self.get_normalised_irradiance().sum(axis=0),
            # cmap=plt.cm.gray,
            extent=[
                np.min(base_coordinates),
                np.max(base_coordinates),
                np.min(base_coordinates),
                np.max(base_coordinates),
            ],
        )
        # plt.colorbar()
        plt.xlabel("Axe x (m)")
        plt.ylabel("Axe y (m)")
        plt.grid(False)
        # plt.show()

    def display_phase(self, colorbar=True):
        base_coordinates = self.get_coordinates(self.wavelengths)
        if np.size(self.wavelengths) > 1:
            plt.suptitle("Phase for each wavelengths", y=0.85, fontsize=16)
            multidimensional_plot(
                self.phase,
                titles_list=[
                    str(wavelength * 1e6) + " µm" for wavelength in self.wavelengths
                ],
                coordinates=base_coordinates,
                cmap=plt.cm.gray,
                colorbar=colorbar,
            )
        else:
            # plt.title("Irradiance of the pupil")

            plt.imshow(
                self.phase[0],  # cmap=plt.cm.gray,
                extent=[
                    np.min(base_coordinates),
                    np.max(base_coordinates),
                    np.min(base_coordinates),
                    np.max(base_coordinates),
                ],
            )
            # plt.colorbar()
            plt.xlabel("Axe x (m)")
            plt.ylabel("Axe y (m)")
            # plt.show()

    def display_apodization(self, colorbar=True):
        base_coordinates = self.get_coordinates(self.wavelengths)
        if np.size(self.wavelengths) > 1:
            plt.suptitle("Apodization for each wavelengths", y=0.85, fontsize=16)
            multidimensional_plot(
                self.apodization,
                titles_list=[
                    str(wavelength * 1e6) + " µm" for wavelength in self.wavelengths
                ],
                coordinates=base_coordinates,
                cmap=plt.cm.gray,
                colorbar=colorbar,
            )
        else:
            plt.title("Apodization of the pupil plane")
            plt.imshow(
                self.apodization[0],
                cmap=plt.cm.gray,
                extent=[
                    np.min(base_coordinates),
                    np.max(base_coordinates),
                    np.min(base_coordinates),
                    np.max(base_coordinates),
                ],
            )
            plt.colorbar()
            plt.show()


class Pupil(Electric_field):
    """Special case of the Electric_field class which englobes what we
    could see at a simple telescope pupil.
    obscuration_percentage is for circular central obscuration,
    aperture is circular."""

    def __init__(self, aperture_diameter=25.4e-3, obscuration_percentage=0, **kwargs):
        super().__init__(**kwargs)
        # Apodization gives the shape of the Pupil
        self.__aperture_diameter = aperture_diameter
        self.apodization = np.select(
            [
                (aperture_diameter * obscuration_percentage) / 2
                > self.get_radius(self.wavelengths),
                self.get_radius(self.wavelengths) <= (aperture_diameter / 2),
            ],
            [0, 1],
        )

    @property
    def aperture_diameter(self):
        return self.__aperture_diameter

    def get_aperture_position(self):
        # w, 4
        # xmin xmax, ymin, ymax
        conditionx = np.sum(self.apodization > 0, axis=(1))
        aperture_x = np.apply_along_axis(np.nonzero, 1, conditionx)[:, 0]
        xmin = np.min(aperture_x, axis=1)
        xmax = np.max(aperture_x, axis=1)
        conditiony = np.sum(self.apodization > 0, axis=(1))
        aperture_y = np.apply_along_axis(np.nonzero, 1, conditiony)[:, 0]
        ymin = np.min(aperture_y, axis=1)
        ymax = np.max(aperture_y, axis=1)
        return np.stack([xmin, xmax + 1, ymin, ymax + 1]).T

    def _get_raw_padded_size(self):
        pos = self.get_aperture_position()
        return np.array([np.diff(pos[:, 0:2]).T, np.diff(pos[:, 2:]).T])[:, 0]

    def get_padded_phase_size(self):
        shapes = self._get_raw_padded_size()
        return np.size(self.wavelengths), *shapes.max(axis=1)

    def set_padded_phase(self, padded_phase):
        wavemax = np.argmax(self._get_raw_padded_size(), axis=1)
        posx = self.get_aperture_position()[:, :2][wavemax[0]]
        posy = self.get_aperture_position()[:, 2:][wavemax[1]]
        temp_phase = self.phase * 0
        temp_phase[:, posx[0] : posx[1], posy[0] : posy[1]] = padded_phase
        self.phase = temp_phase

    def get_normalised_radius(self):
        return self.get_radius(self.wavelengths) * 2 / self.__aperture_diameter

    def phase_change_noll(self, indices, weights=None):
        try:
            indices[1]
            indices = np.array(indices)
        except TypeError or IndexError:
            indices = np.array((indices,))
        pup_radius = self.get_normalised_radius()
        pup_angle = self.get_angle(self.wavelengths)
        if len(indices.shape) == 2:
            self.phase = np.array(
                [
                    noll_zernike(
                        indices[index],
                        pup_radius[index],
                        pup_angle[index],
                        weights[index],
                    )
                    for index in range(indices.shape[0])
                ]
            )
        elif len(indices.shape) == 1:
            self.phase = noll_zernike(indices, pup_radius, pup_angle, weights)

    def phase_add_noll(self, indices, weights=None):
        try:
            indices[1]
            indices = np.array(indices)
        except TypeError or IndexError:
            indices = np.array((indices,))
        pup_radius = self.get_normalised_radius()
        pup_angle = self.get_angle(self.wavelengths)
        if len(indices.shape) == 2:
            self.phase += np.array(
                [
                    noll_zernike(
                        indices[index], pup_radius[index], pup_angle[index], weights
                    )
                    for index in range(indices.shape[0])
                ]
            )
        elif len(indices.shape) == 1:
            self.phase += noll_zernike(indices, pup_radius, pup_angle, weights)

    def phase_change_zernike(self, indices, weights=None):
        self.phase = zernike(
            indices,
            self.get_normalised_radius(),
            self.get_angle(self.wavelengths),
            weights,
        )

    def phase_change_fourier(self, indices, weights=None):
        """Frequency given by the index in groups of 4.
        index//4 -> frequency (multiples ).
        index%4 -> function: (1,2,3,4) -> (sin(x),cos(x),sin(y),cos(y))"""
        try:
            indices[1]
            indices = np.array(indices)
        except TypeError or IndexError:
            indices = np.array((indices,))
        pup_norm_coordinates = (
            self.get_coordinates(self.wavelengths) * 2 / self.__aperture_diameter
        )
        if len(indices.shape) == 2:
            self.phase = np.array(
                [
                    fourier(
                        indices[index],
                        pup_norm_coordinates[index][1],
                        pup_norm_coordinates[index][0],
                        weights,
                    )
                    for index in range(indices.shape[0])
                ]
            )
        elif len(indices.shape) == 1:
            self.phase = fourier(
                indices,
                pup_norm_coordinates[:, 0, :, :],
                pup_norm_coordinates[:, 1, :, :],
                weights,
            )

    def get_psf(self, propagating_function):
        """This function needs a propagator function in order to compute
        the PSF"""
        return abs(propagating_function(self.get_field())) ** 2

    def display_psf(self, propagating_function, colorbar=True):
        """This display function need a propagator in order to compute
        the PSF"""
        prop_coordinates = self.get_propagated_coordinates(10)
        if np.size(self.wavelengths) > 1:
            plt.suptitle("PSF for each wavelengths", y=0.85, fontsize=16)
            multidimensional_plot(
                self.get_psf(propagating_function),
                titles_list=[
                    str(wavelength * 1e6) + " µm" for wavelength in self.wavelengths
                ],
                coordinates=prop_coordinates,
                cmap=plt.cm.gray,
                colorbar=colorbar,
            )
        else:
            plt.title("PSF at {}".format(str(self.wavelengths * 1e6) + " µm"))
            plt.imshow(
                self.get_psf(propagating_function)[0],
                cmap=plt.cm.gray,
                extent=[
                    np.min(prop_coordinates, axis=(1, 2, 3)),
                    np.max(prop_coordinates, axis=(1, 2, 3)),
                    np.min(prop_coordinates, axis=(1, 2, 3)),
                    np.max(prop_coordinates, axis=(1, 2, 3)),
                ],
            )
            plt.colorbar()
            plt.show()
