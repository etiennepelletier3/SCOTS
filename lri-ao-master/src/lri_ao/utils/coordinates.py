#!/usr/bin/env python

import numpy as np


class Coordinate_system:

    """Given a coordinate system (pupil or image) perform the necessary
    manipulations in order to create a coherant coordinate_system.
    The cartesian coordinates are calculated with numpy's ogrid.

    You can also initialise this class using a init_coordinates that has
    the shape (n,2,sampling,sampling). This will override most of the class
    properties as well."""

    def __init__(
        self,
        sampling_points=501,
        side_length=50e-3,
        init_coordinates=0,
        line_array=False,
    ):
        self.__sampling_points = sampling_points
        self.__side_length = side_length
        self.__line_array = line_array
        test_init_coordinates_shape = np.array(init_coordinates).shape

        if np.size(test_init_coordinates_shape) == 0:
            if self.__line_array is True:
                self.__coordinates = np.linspace(
                    -self.side_length / 2, self.side_length / 2, self.sampling_points
                ).reshape((1, self.sampling_points))
                # print('line',self.__coordinates.shape)
                self.__coordinates = np.mgrid[
                    -self.side_length
                    / 2 : self.side_length
                    / 2 : self.sampling_points
                    * 1j,
                    -0 : 0 : 1 * 1j,
                ]
                # print('array',self.__coordinates.shape)
            else:
                self.__coordinates = np.mgrid[
                    -self.side_length
                    / 2 : self.side_length
                    / 2 : self.sampling_points
                    * 1j,
                    -self.side_length
                    / 2 : self.side_length
                    / 2 : self.sampling_points
                    * 1j,
                ]
        elif test_init_coordinates_shape[0] >= 1:
            self.__coordinates = init_coordinates
        else:
            raise (ValueError("init_coordinate cannot be parsed"))
        self.__sampling_points = self.__coordinates.shape[-2]
        if self.__coordinates.shape[-1] == 1:
            self.__line_array = True

    @property
    def sampling_points(self):
        return self.__sampling_points

    @property
    def line_array(self):
        return self.__line_array

    @property
    def side_length(self):
        return self.__side_length

    def set_coordinates(self, new_coordinates):
        self.__coordinates = new_coordinates

    def get_coordinates(self, wavelengths=0):
        """Return base wavelength defined by the class init"""
        if np.size(self.__coordinates.shape) == 3:
            if self.__line_array is True:
                # print(self.__coordinates)
                # print(self.__sampling_points)
                return np.rollaxis(
                    np.repeat(
                        self.__coordinates.reshape(2, self.sampling_points, 1, 1),
                        np.size(wavelengths),
                    ).reshape(2, self.sampling_points, 1, np.size(wavelengths)),
                    -1,
                )
            else:
                return np.rollaxis(
                    np.repeat(
                        self.__coordinates.reshape(
                            2, self.sampling_points, self.sampling_points, 1
                        ),
                        np.size(wavelengths),
                    ).reshape(
                        2,
                        self.sampling_points,
                        self.sampling_points,
                        np.size(wavelengths),
                    ),
                    -1,
                )
        else:
            return self.__coordinates

    def get_radius(self, wavelengths=0):
        return np.sqrt(
            self.get_coordinates(wavelengths=wavelengths)[:, 0, :, :] ** 2
            + self.get_coordinates(wavelengths=wavelengths)[:, 1, :, :] ** 2
        )

    def get_angle(self, wavelengths=0):
        return np.arctan2(
            self.get_coordinates(wavelengths=wavelengths)[:, 0, :, :],
            self.get_coordinates(wavelengths=wavelengths)[:, 1, :, :],
        )

    def get_polar_coordinates(self, wavelengths=0):
        return np.swapaxes(
            np.array(
                [
                    self.get_radius(wavelengths=wavelengths),
                    self.get_angle(wavelengths=wavelengths),
                ]
            ),
            0,
            1,
        )
