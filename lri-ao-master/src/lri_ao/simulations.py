import numpy as np
from copy import deepcopy

from scipy.ndimage.interpolation import zoom
from .optics.bank import Dummy_element
from .propagators import fourier_transform_propagator
from .fields import Electric_field, Pupil
from .sensors import CCD_camera


def photon_budget_capella(magnitude=[1, 1]):
    """Number of photon calculation per second for the three bands in the
    visible. Quite limited, but OK for these simulations.
    https://www.cfa.harvard.edu/~dfabricant/huchra/ay145/mags.html
    # B V and R bands"""
    magnitude = np.array(magnitude)
    flux = 1.51e7 * np.array([(0.16 * 3640), (0.23 * 3080)])
    flux_mag = flux * 10 ** (-magnitude / 2.5)
    diam = 0.3556
    aire = np.pi * (diam / 2) ** 2
    # flux * aire * obscuration centrale *
    # 0.835 [Celestron system transmission] * (system transmission)
    nb_photon = np.floor(flux_mag * aire * 0.9 * 0.835 * 0.98**15 * 0.96**3).astype(
        int
    )
    return nb_photon


class AO_simulation_coherent:
    """Parent class for AO_simulations using Pupil as first plane. Uses
    focal_length argument for propagation and optical_element_list to
    apply optical_element phase_shift to the propagation. Each change
    in the pupil_plane must be followed by an update."""

    def __init__(
        self,
        focal_length_list=100e-3,
        wavelengths=640,
        sampling_points=501,
        side_length=50e-3,
        init_coordinates=0,
        aperture_diameter=3e-3,
        obscuration_percentage=0,
        optical_element_list=Dummy_element(),
        photon_budget=1e8,
        propagator=fourier_transform_propagator,
        mag=1,
        line_array=False,
        CCD_camera_object=CCD_camera(),
        **kwargs
    ):

        self.__CCD_camera = CCD_camera_object
        self.__photon_budget = np.array(photon_budget)
        self.__pupil_plane = Pupil(
            wavelengths=wavelengths,
            sampling_points=sampling_points,
            side_length=side_length,
            obscuration_percentage=obscuration_percentage,
            aperture_diameter=aperture_diameter,
            line_array=line_array,
            photon_budget=self.photon_budget,
        )

        self.__propagator = fourier_transform_propagator
        self.__fourier_planes = []
        self.unshifted_fourier_planes = []
        self.mag = mag

        # Parsing of arguments
        if np.size(focal_length_list) != np.size(optical_element_list):
            raise (ValueError("Must have same number of optical_element"))
        if np.size(focal_length_list) <= 1:
            self.__focal_length_list = [
                focal_length_list,
            ]
            self.__optical_element_list = [
                optical_element_list,
            ]
        else:
            self.__focal_length_list = focal_length_list
            self.__optical_element_list = optical_element_list
        # Testing if we need to generate electricfield

        if not np.all((focal_length_list, optical_element_list) == np.array(None)):
            # Generation of Electric_field
            (
                self.unshifted_fourier_planes.append(
                    Electric_field(
                        wavelengths=self.wavelengths,
                        init_coordinates=self.pupil_plane.get_propagated_coordinates(
                            self.focal_length_list[0]
                        ),
                        sampling_points=sampling_points,
                        side_length=side_length,
                        line_array=line_array,
                        photon_budget=self.photon_budget,
                    )
                )
            )
            (
                self.fourier_planes.append(
                    Electric_field(
                        wavelengths=self.wavelengths,
                        init_coordinates=self.pupil_plane.get_propagated_coordinates(
                            self.focal_length_list[0]
                        ),
                        sampling_points=sampling_points,
                        side_length=side_length,
                        line_array=line_array,
                        photon_budget=self.photon_budget,
                    )
                )
            )
            for i in range(np.size(focal_length_list) - 1):
                (
                    self.unshifted_fourier_planes.append(
                        Electric_field(
                            wavelengths=self.wavelengths,
                            init_coordinates=self.fourier_planes[
                                i
                            ].get_propagated_coordinates(self.focal_length_list[i + 1]),
                            sampling_points=sampling_points,
                            side_length=side_length,
                            line_array=line_array,
                            photon_budget=self.photon_budget,
                        )
                    )
                )
                (
                    self.fourier_planes.append(
                        Electric_field(
                            wavelengths=self.wavelengths,
                            init_coordinates=self.fourier_planes[
                                i
                            ].get_propagated_coordinates(self.focal_length_list[i + 1]),
                            sampling_points=sampling_points,
                            side_length=side_length,
                            line_array=line_array,
                        )
                    )
                )
        # Generation of induced field list to be calculated once
        # 0 being pupil_plane
        self.update_induced_field_list()
        # self.update()

    @property
    def propagator(self):
        return self.__propagator

    @property
    def photon_budget(self):
        return self.__photon_budget

    @property
    def CCD_camera(self):
        return self.__CCD_camera

    @property
    def wavelengths(self):
        return self.pupil_plane.wavelengths

    @property
    def focal_length_list(self):
        return self.__focal_length_list

    @property
    def pupil_plane(self):
        return self.__pupil_plane

    @property
    def fourier_planes(self):
        return self.__fourier_planes

    @property
    def optical_element_list(self):
        return self.__optical_element_list

    @property
    def induced_field_list(self):
        return self.__induced_field_list

    def get_plane_polar_coordinates(self, index):
        if index == 0:
            return self.pupil_plane.get_polar_coordinates(self.wavelengths)
        else:
            return self.fourier_planes[index].get_polar_coordinates(self.wavelengths)

    def get_plane_coordinates(self, index):
        if index == 0:
            return self.pupil_plane.get_coordinates(self.wavelengths)
        else:
            return self.fourier_planes[index].get_coordinates(self.wavelengths)

    def get_element_sag(self, index):
        if index == 0:
            return self.optical_element_list[0].get_sag(
                self.pupil_plane.get_polar_coordinates(self.wavelengths),
                self.wavelengths,
            )
        if index > 0:
            return self.optical_element_list[index + 1].get_sag(
                self.fourier_planes[index].get_polar_coordinates(self.wavelengths),
                self.wavelengths,
            )

    def get_element_phase(self, index):
        if index == 0:
            return self.optical_element_list[0].get_phase(
                self.pupil_plane.get_polar_coordinates(self.wavelengths),
                self.wavelengths,
            )
        if index > 0:
            return self.optical_element_list[index + 1].get_phase(
                self.fourier_planes[index].get_polar_coordinates(self.wavelengths),
                self.wavelengths,
            )

    def update_induced_field_list(self):
        if np.size(self.fourier_planes) > 0:
            self.__induced_field_list = []
            (
                self.__induced_field_list.append(
                    self.optical_element_list[0].get_induced_phase_shift(
                        self.pupil_plane.get_polar_coordinates(self.wavelengths),
                        self.wavelengths,
                    )
                )
            )
            for plane in range(np.size(self.focal_length_list) - 1):
                if plane & 1:
                    (
                        self.__induced_field_list.append(
                            self.optical_element_list[
                                plane + 1
                            ].get_induced_phase_shift(
                                self.fourier_planes[plane].get_polar_coordinates(
                                    self.wavelengths
                                ),
                                self.wavelengths,
                            )
                        )
                    )
                else:
                    self.__induced_field_list.append(
                        np.conj(
                            self.optical_element_list[
                                plane + 1
                            ].get_induced_phase_shift(
                                self.fourier_planes[plane].get_polar_coordinates(
                                    self.wavelengths
                                ),
                                self.wavelengths,
                            )
                        )
                    )

    def get_element_phase_shift(self, index):
        return self.optical_element_list[0].get_induced_phase_shift(
            self.pupil_plane.get_polar_coordinates(self.wavelengths), self.wavelengths
        )

    def update(self):
        if np.size(self.fourier_planes) > 0:
            field_temp = self.propagator(
                self.pupil_plane.get_field() * self.induced_field_list[0], shift=True
            )
            for plane in range(np.size(self.fourier_planes) - 1):
                if plane & 1:
                    self.fourier_planes[plane].update(field_temp)
                    self.unshifted_fourier_planes[plane].update(field_temp)
                    field_temp = self.propagator(
                        self.unshifted_fourier_planes[plane].get_field()
                        * self.induced_field_list[plane + 1],
                        shift=True,
                    )
                else:
                    self.fourier_planes[plane].update(field_temp[1])
                    self.unshifted_fourier_planes[plane].update(field_temp[0])
                    field_temp = self.propagator(
                        self.unshifted_fourier_planes[plane].get_field()
                        * self.induced_field_list[plane + 1]
                    )
            if np.size(self.fourier_planes) & 1:
                self.fourier_planes[-1].update(field_temp[1])
                self.unshifted_fourier_planes[-1].update(field_temp[0])
            else:
                self.fourier_planes[-1].update(field_temp)

    @property
    def detector_plane(self):
        if np.size(self.fourier_planes) > 0:
            return self.fourier_planes[-1]
        else:
            return self.pupil_plane

    def display_detector_total_irradiance(self):
        self.detector_plane.display_total_irradiance()

    def detector_read(self):
        samp = (
            self.pupil_plane.sampling_points // 2
            - self.pupil_plane.sampling_points // (2 * self.mag)
        )
        if samp == 0:
            return self.CCD_camera.read(
                self.detector_plane.get_normalised_irradiance(),
                self.wavelengths,
            )
        else:
            return self.CCD_camera.read(
                self.detector_plane.get_normalised_irradiance()[
                    :, samp : samp * -1, samp : samp * -1
                ],
                self.wavelengths,
            )


class AO_simulation_coherent_shs(AO_simulation_coherent):
    """Special case of AO_simulations_coherent where the SHS sensor must
    be placed a the detector plane. As such, the __init__ of this class
    test this first. A special operation is made to speed up the shs
    calculation and imposes retriction on it. In fact:

    aperture_diameter/side_length < min(wavelength)/max(wavelength)

    If this is respected, everything should work as intentended."""

    def __init__(
        self,
        nb_cote_ulentilles=35,
        fnumb_ulentilles=50,
        shs_focal_length=5e-3,
        side_length=50e-3,
        aperture_diameter=25.4e-3,
        wavelengths=640e-9,
        sampling_points=500,
        **kwargs
    ):
        self.__fnumb_ulentilles = fnumb_ulentilles
        self.__init = True
        self.__nb_cote_ulentilles = nb_cote_ulentilles
        self.__natural_focal = (side_length / nb_cote_ulentilles) ** 2 / (
            ((sampling_points / nb_cote_ulentilles) - 1) * np.max(wavelengths)
        )
        super().__init__(
            side_length=side_length,
            aperture_diameter=aperture_diameter,
            wavelengths=wavelengths,
            sampling_points=sampling_points,
            **kwargs
        )
        if np.size(self.focal_length_list) == 0:
            self.detector_plane_electric_field = self.pupil_plane
        else:
            self.detector_plane_electric_field = self.fourier_planes[-1]
        # if (aperture_diameter/side_length <
        #    np.min(wavelengths)/np.max(wavelengths)):
        #     raise(ValueError("Ratio aperture/side_length must be bigger\
        #                       than wavelengths"))
        if np.size(self.focal_length_list) % 2 != 0:
            raise (ValueError("Must have pupil space on detector plane"))
        self.__init = False
        self.real_focal = self.fnumb_ulentilles * (
            np.max(self.detector_plane_electric_field.get_coordinates())
            / self.nb_cote_ulentilles
        )
        self.__shs_real_focal = shs_focal_length
        # self.update()

    @property
    def micro_lens_plane_electric_field(self):
        return self.__micro_lens_plane_electric_field

    @property
    def shs_real_focal(self):
        return self.__shs_real_focal

    @property
    def natural_focal(self):
        return self.__natural_focal

    @property
    def fnumb_ulentilles(self):
        return self.__fnumb_ulentilles

    @property
    def nb_cote_ulentilles(self):
        return self.__nb_cote_ulentilles

    def update(self):
        super().update()
        if not self.__init:
            self.__micro_lens_plane_electric_field = deepcopy(
                self.detector_plane_electric_field
            )
            micro_lens_array_detector_field = (
                self.detector_plane_electric_field.get_zoomed_field()
            )
            micro_lens_array_size = (
                micro_lens_array_detector_field.shape[-1] / self.nb_cote_ulentilles
            )

            # split detector field
            detector_field_split_1 = np.array(
                np.array_split(
                    zoom(
                        micro_lens_array_detector_field,
                        (
                            micro_lens_array_detector_field.shape[-1]
                            // self.nb_cote_ulentilles
                        )
                        / micro_lens_array_size,
                    ),
                    self.nb_cote_ulentilles,
                    axis=1,
                )
            )
            full_split_field = np.array(
                np.split(detector_field_split_1, self.nb_cote_ulentilles, axis=3)
            )

            # Where is the light significant?
            sum_inside_microlens = abs(full_split_field.sum(axis=(2, 3, 4)))
            computed_micro_lenses_boolean = ~np.isclose(
                sum_inside_microlens, np.zeros_like(sum_inside_microlens)
            )
            computed_micro_lenses_number = computed_micro_lenses_boolean[
                computed_micro_lenses_boolean
            ].shape[0]

            computed_micro_lenses_indices = np.indices(
                computed_micro_lenses_boolean.shape
            )[:, computed_micro_lenses_boolean]
            mul_factor = 2
            zoom_factor = 1
            computed_full_field_indices = np.zeros(
                (
                    2,
                    computed_micro_lenses_indices.shape[1],
                    (full_split_field.shape[-1] * zoom_factor) ** 2,
                ),
                dtype=int,
            )
            # computed_final_field_indices = np.zeros_like(computed_full_field_indices)
            computed_full_field_indices[:, :, :] = (
                np.tile(
                    np.indices(
                        np.array(full_split_field.shape[3:]) * zoom_factor
                    ).reshape(2, -1),
                    computed_micro_lenses_indices.shape[-1],
                ).reshape(computed_full_field_indices.shape)
                + (((full_split_field.shape[3] * zoom_factor * (mul_factor - 1))) // 2)
                # + indices.reshape(2, -1, 1) * splitsplit.shape[-1]
            )

            # Init Electric_field before propagation
            detector_field_temp = Electric_field(
                wavelengths=np.tile(self.wavelengths, computed_micro_lenses_number),
                side_length=self.detector_plane.side_length
                * mul_factor
                / self.nb_cote_ulentilles,
                sampling_points=full_split_field.shape[3] * mul_factor * zoom_factor,
            )

            pre_propagation = np.zeros(
                (
                    computed_micro_lenses_number,
                    full_split_field.shape[2],
                    full_split_field.shape[3] * mul_factor * zoom_factor,
                    full_split_field.shape[4] * mul_factor * zoom_factor,
                ),
                dtype=complex,
            )
            # print(pre_propagation.shape)
            all_images = full_split_field[
                computed_micro_lenses_indices[0], computed_micro_lenses_indices[1]
            ]

            for wavelength in range(pre_propagation.shape[1]):
                for subimage in range(pre_propagation.shape[0]):
                    pre_propagation[
                        subimage,
                        wavelength,
                        computed_full_field_indices[1].T[:, subimage],
                        computed_full_field_indices[0].T[:, subimage],
                    ] = zoom(
                        all_images[subimage, wavelength],
                        zoom_factor,
                    ).flatten()

            pre_propagation = pre_propagation.reshape(
                -1, pre_propagation.shape[-2], pre_propagation.shape[-1]
            )
            post_propagation = np.empty_like(pre_propagation)
            for x in range(pre_propagation.shape[0]):
                post_propagation[x] = np.fft.fftshift(
                    self.propagator(pre_propagation[x])
                )

            side_length_last_plane = (
                detector_field_temp.get_propagated_coordinates(self.shs_real_focal)[
                    :, -1, 0
                ].max(axis=-1)
                * 2
            )

            final_field = np.zeros(
                (
                    full_split_field.shape[2],
                    full_split_field.shape[0] * full_split_field.shape[3],
                    full_split_field.shape[1] * full_split_field.shape[4],
                ),
                dtype=complex,
            )
            prepared_array = post_propagation.reshape(
                -1,
                full_split_field.shape[2],
                pre_propagation.shape[-2],
                pre_propagation.shape[-1],
            )
            zoomfactor = (
                side_length_last_plane.reshape(
                    -1,
                    prepared_array.shape[1],
                )
                / detector_field_temp.sampling_points
            ) / (self.detector_plane.side_length / self.detector_plane.sampling_points)

            for wavelength in range(prepared_array.shape[1]):
                for subimage in range(prepared_array.shape[0]):
                    loc_subimage = zoom(
                        abs(prepared_array[subimage, wavelength, :, :]),
                        zoomfactor[subimage, wavelength],
                    )
                    loc_index = np.clip(
                        (
                            computed_micro_lenses_indices[:, subimage]
                            * full_split_field.shape[-1]
                            + full_split_field.shape[-1] // 2
                        ).reshape(-1, 1, 1)
                        + np.indices(loc_subimage.shape)
                        - loc_subimage.shape[0] // 2,
                        0,
                        final_field.shape[-1] - 1,
                    ).reshape(2, -1)
                    final_field[
                        wavelength, loc_index[1], loc_index[0]
                    ] += loc_subimage.flatten()
            if final_field.shape != self.detector_plane_electric_field.phase.shape:
                zoomed_final_field = np.empty_like(
                    self.detector_plane_electric_field.phase, dtype=complex
                )
                for wav in range(zoomed_final_field.shape[0]):
                    zoomed_final_field[wav] = zoom(
                        final_field[wav],
                        self.detector_plane_electric_field.phase.shape[-1]
                        / final_field.shape[-1],
                    )
            else:
                zoomed_final_field = final_field
            self.detector_plane_electric_field.update(zoomed_final_field)

    @property
    def detector_plane(self):
        return self.detector_plane_electric_field


class AO_simulation_coherent_modulating_pyr(AO_simulation_coherent):
    def __init__(self, modulation_width_pixels=0, nb_modulation=0, **kwargs):
        self.modulation_width_pixels = np.array(modulation_width_pixels)
        self.nb_modulation = nb_modulation
        super().__init__(**kwargs)

    def update(self):
        # if np.size(self.fourier_planes) > 0:
        #     field_temp = (self.propagator(
        #                   self.pupil_plane.get_field() *
        #                   self.induced_field_list[0], shift=True))
        #     for plane in range(np.size(self.fourier_planes)-1):
        #         # self.fourier_planes[plane].update(field_temp)
        #         # field_temp = (self.propagator(
        #         #               self.fourier_planes[plane].get_field() *
        #         #               self.induced_field_list[plane+1]))
        #         if plane&1:
        #             self.fourier_planes[plane].update(field_temp)
        #             self.unshifted_fourier_planes[plane].update(field_temp)
        #             field_temp = (self.propagator(
        #                           self.unshifted_fourier_planes[plane].get_field() *
        #                           self.induced_field_list[plane+1], shift=True))
        #         else:
        #             self.fourier_planes[plane].update(field_temp[1])
        #             self.unshifted_fourier_planes[plane].update(field_temp[0])
        #             field_temp = (self.propagator(
        #                           self.unshifted_fourier_planes[plane].get_field() *
        #                           self.induced_field_list[plane+1]))
        #     if np.size(self.fourier_planes)&1:
        #         self.fourier_planes[-1].update(field_temp[1])
        #         self.unshifted_fourier_planes[-1].update(field_temp[0])
        #     else:
        #         self.fourier_planes[-1].update(field_temp)

        self.modulation_width_pixels = (
            self.modulation_width_pixels * np.max(self.wavelengths) / self.wavelengths
        )
        if np.size(self.fourier_planes) > 0:
            field_temp = self.propagator(
                self.pupil_plane.get_field() * self.induced_field_list[0], shift=True
            )
            for plane in range(np.size(self.fourier_planes) - 1):
                if (
                    plane == np.size(self.fourier_planes) - 2
                    and self.nb_modulation != 0
                ):
                    if plane & 1:
                        try:
                            field_temp_sum = np.swapaxes(
                                np.array(
                                    [
                                        [
                                            np.roll(
                                                field_temp[a],
                                                (
                                                    np.int(
                                                        self.modulation_width_pixels[a]
                                                        * np.cos(t)
                                                    ),
                                                    np.int(
                                                        self.modulation_width_pixels[a]
                                                        * np.sin(t)
                                                    ),
                                                ),
                                                (0, 1),
                                            )
                                            for t in np.linspace(
                                                0,
                                                2 * np.pi,
                                                self.nb_modulation,
                                                endpoint=False,
                                            )
                                        ]
                                        for a in range(np.shape(self.wavelengths)[0])
                                    ]
                                ),
                                0,
                                1,
                            )
                        except Exception:
                            field_temp_sum = [
                                np.roll(
                                    field_temp,
                                    (
                                        np.int(
                                            self.modulation_width_pixels * np.cos(t)
                                        ),
                                        np.int(
                                            self.modulation_width_pixels * np.sin(t)
                                        ),
                                    ),
                                    (1, 2),
                                )
                                for t in np.linspace(
                                    0, 2 * np.pi, self.nb_modulation, endpoint=False
                                )
                            ]
                        self.fourier_planes[plane].update(
                            np.sum(np.abs(field_temp_sum), axis=0)
                        )
                        field_irr = []
                        for field_temp_ele in field_temp_sum:
                            field_temp = self.propagator(
                                field_temp_ele * self.induced_field_list[plane + 1],
                                shift=True,
                            )
                            field_irr.append(np.abs(field_temp) ** 2)
                        field_temp = np.sqrt(np.sum(field_irr, axis=0))
                    else:
                        field_temp_norm = field_temp[1]
                        field_temp_shift = field_temp[0]
                        try:
                            field_temp_sum_norm = np.swapaxes(
                                np.array(
                                    [
                                        [
                                            np.roll(
                                                field_temp_norm[a],
                                                (
                                                    np.int(
                                                        self.modulation_width_pixels[a]
                                                        * np.cos(t)
                                                    ),
                                                    np.int(
                                                        self.modulation_width_pixels[a]
                                                        * np.sin(t)
                                                    ),
                                                ),
                                                (0, 1),
                                            )
                                            for t in np.linspace(
                                                0,
                                                2 * np.pi,
                                                self.nb_modulation,
                                                endpoint=False,
                                            )
                                        ]
                                        for a in range(np.shape(self.wavelengths)[0])
                                    ]
                                ),
                                0,
                                1,
                            )
                            field_temp_sum_shift = np.swapaxes(
                                np.array(
                                    [
                                        [
                                            np.roll(
                                                field_temp_shift[a],
                                                (
                                                    np.int(
                                                        self.modulation_width_pixels[a]
                                                        * np.cos(t)
                                                    ),
                                                    np.int(
                                                        self.modulation_width_pixels[a]
                                                        * np.sin(t)
                                                    ),
                                                ),
                                                (0, 1),
                                            )
                                            for t in np.linspace(
                                                0,
                                                2 * np.pi,
                                                self.nb_modulation,
                                                endpoint=False,
                                            )
                                        ]
                                        for a in range(np.shape(self.wavelengths)[0])
                                    ]
                                ),
                                0,
                                1,
                            )
                        except Exception:
                            field_temp_sum_norm = [
                                np.roll(
                                    field_temp_norm,
                                    (
                                        np.int(
                                            self.modulation_width_pixels * np.cos(t)
                                        ),
                                        np.int(
                                            self.modulation_width_pixels * np.sin(t)
                                        ),
                                    ),
                                    (1, 2),
                                )
                                for t in np.linspace(
                                    0, 2 * np.pi, self.nb_modulation, endpoint=False
                                )
                            ]
                            field_temp_sum_shift = [
                                np.roll(
                                    field_temp_shift,
                                    (
                                        np.int(
                                            self.modulation_width_pixels * np.cos(t)
                                        ),
                                        np.int(
                                            self.modulation_width_pixels * np.sin(t)
                                        ),
                                    ),
                                    (1, 2),
                                )
                                for t in np.linspace(
                                    0, 2 * np.pi, self.nb_modulation, endpoint=False
                                )
                            ]
                        self.fourier_planes[plane].update(
                            np.sum(np.abs(field_temp_sum_norm), axis=0)
                        )
                        self.unshifted_fourier_planes[plane].update(
                            np.sum(np.abs(field_temp_sum_shift), axis=0)
                        )
                        field_irr = []
                        for field_temp_ele in field_temp_sum_shift:
                            field_temp = self.propagator(
                                field_temp_ele * self.induced_field_list[plane + 1]
                            )
                            field_irr.append(np.abs(field_temp) ** 2)
                        field_temp = np.sqrt(np.sum(field_irr, axis=0))

                else:
                    if plane & 1:
                        self.fourier_planes[plane].update(field_temp)
                        self.unshifted_fourier_planes[plane].update(field_temp)
                        field_temp = self.propagator(
                            self.unshifted_fourier_planes[plane].get_field()
                            * self.induced_field_list[plane + 1],
                            shift=True,
                        )
                    else:
                        self.fourier_planes[plane].update(field_temp[1])
                        self.unshifted_fourier_planes[plane].update(field_temp[0])
                        field_temp = self.propagator(
                            self.unshifted_fourier_planes[plane].get_field()
                            * self.induced_field_list[plane + 1]
                        )
            if np.size(self.fourier_planes) & 1:
                self.fourier_planes[-1].update(field_temp[1])
                self.unshifted_fourier_planes[-1].update(field_temp[0])
            else:
                self.fourier_planes[-1].update(field_temp)
