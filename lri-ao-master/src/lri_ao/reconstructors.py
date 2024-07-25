import warnings
import numpy as np
from scipy.signal import find_peaks
from .utils import noll_gradZernike, zoom, noll_zernike, cog


def SHS_AUTO_INIT(
    input_image: np.ndarray,
    thresh: float = 0.35,
    ml_pitch=None,
    mask=None,
    image_slices=None,
    **kwargs,
):
    if input_image.ndim != 2:
        raise ValueError("input_image must be a 2D array")
    # compute fft
    if ml_pitch is None:
        ffta = abs(np.fft.fftshift(np.fft.fft2(input_image))) ** 2
        ffta[ffta.shape[0] // 2, ffta.shape[1] // 2] = 0
        fftax = ffta[ffta.shape[0] // 2]
        fftay = ffta[:, ffta.shape[0] // 2]
        fftfreqx = np.fft.fftshift(np.fft.fftfreq(fftax.shape[0]))
        fftfreqy = np.fft.fftshift(np.fft.fftfreq(fftay.shape[0]))
        # find peaks in in fft
        peaksx, _ = find_peaks(fftax, distance=5)
        peaksy, _ = find_peaks(fftay)
        x_ml_pitch = 1 / abs(fftfreqx[peaksx[np.argmax(fftax[peaksx])]])
        y_ml_pitch = 1 / abs(fftfreqy[peaksy[np.argmax(fftay[peaksy])]])
        ml_pitch = (x_ml_pitch + y_ml_pitch) / 2

    image_resolution = input_image.shape
    shs_object = SHS_reconstructor(
        mask.shape
        if np.any(mask)
        else np.floor(image_resolution / ml_pitch).astype(int),
        image_resolution,
        mask=mask,
        image_slices=image_slices,
        **kwargs,
    )

    if mask is None:
        shs_object.reconstruction_zonal(input_image)
        mask_prim = shs_object.splitted_image.mean(axis=(3, 2))
        mask_prim -= mask_prim.min()
        mask_prim /= mask_prim.max()
        mask_prim = mask_prim > thresh

        shs_object.mask = mask_prim
        slices_num = [
            [idx.min(), (idx.max() + 1)] for idx in np.nonzero(shs_object.mask)
        ]
        if slices_num != [[0, x] for x in mask_prim.shape]:
            image_slices = tuple(
                slice(*np.floor(np.array(x) * ml_pitch).astype(int)) for x in slices_num
            )
            shs_object = SHS_AUTO_INIT(
                input_image,
                thresh=thresh,
                image_slices=image_slices,
                ml_pitch=ml_pitch,
                mask=mask_prim[tuple(slice(*x) for x in slices_num)],
                **kwargs,
            )

    return shs_object


class SHS_reconstructor:
    """Class that takes shack-hartmann images and given the number of
    microlenses, their focal length, and the shape of the mask (0 or 1 2d array
    of the size of microlenses), can give
    the abberation of the wavefront incident on the sensor.

    Zonal reconstructor from poppy:
        https://github.com/spacetelescope/poppy/blob/develop/poppy/sub_sampled_optics.py
    """

    def __init__(
        self,
        microlenses_nb: int or tuple,
        image_resolution: int or tuple,
        microlenses_pitch: int = 1,
        microlenses_focal: int = 1,
        reconstruction_radius: float = 1,
        maxmode: int = 20,
        mask: np.ndarray = None,
        image_slices=None,
    ):
        self.__microlenses_nb = np.array(microlenses_nb)
        self.__microlenses_nb = (
            self.__microlenses_nb
            if self.__microlenses_nb.ndim != 2
            else np.tile(self.__microlenses_nb, 2)
        )
        self.__maxmode = maxmode
        self.__microlenses_focal = microlenses_focal
        self.__y, self.__x = np.mgrid[
            -1 : 1 : int(self.__microlenses_nb[0]) * 1j,
            -1 : 1 : int(self.__microlenses_nb[1]) * 1j,
        ]
        self.__r = np.sqrt(self.__y**2 + self.__x**2)
        # self.__r /= reconstruction_radius
        self.__ang = np.arctan2(self.__y, self.__x)
        self.__microlenses_pitch = microlenses_pitch
        self.__image_resolution = np.array(image_resolution)
        self.__image_resolution = (
            self.__image_resolution
            if self.__image_resolution.ndim != 2
            else np.tile(self.__image_resolution, 2)
        )
        self.__image_slices = (
            image_slices
            if image_slices is not None
            else tuple(slice(0, x) for x in self.__image_resolution)
        )
        self.__n_pixel_subarray = self.image_resolution // self.microlenses_nb
        self.__inverse_interference_matrix = None
        # X and Y COG
        # if self.microlenses_nb > 1:
        self.__cogs = np.mgrid[
            slice(0, self.__n_pixel_subarray[0], 1),
            slice(0, self.__n_pixel_subarray[1], 1),
        ].reshape(2, 1, self.__n_pixel_subarray[0], self.__n_pixel_subarray[1])
        self.__cogs -= self.__n_pixel_subarray.reshape(-1, 1, 1, 1) // 2
        # Mask
        self.__flat = 0
        self.__current_image = np.zeros(self.image_resolution)
        self.__splitted_image = np.zeros(self.microlenses_nb)

        if np.all(mask) is None:
            self.__mask = np.ones(self.microlenses_nb).astype(int)
            self.auto_mask = True if np.all(self.microlenses_nb > 1) else False

        else:
            self.__mask = mask
            self.auto_mask = False
            self.__compute_inverse_interference_matrix()
            self.__compute_zonal_matrices()

    @property
    def mode_max(self):
        return self.__maxmode

    @property
    def microlenses_pitch(self):
        return self.__microlenses_pitch

    @property
    def image_slices(self):
        return self.__image_slices

    @property
    def current_image(self):
        return self.__current_image[self.image_slices]

    @property
    def mask(self):
        return self.__mask

    @property
    def r(self):
        return self.__r

    @property
    def ang(self):
        return self.__ang

    @mask.setter
    def mask(self, mask):
        if type(mask) is not np.ndarray:
            mask = np.array(mask)
        if len(mask.shape) != 2:
            raise ValueError("Wrong dimensions for mask")
        # elif mask.shape[0] != mask.shape[1]:
        #     raise ValueError("Wrong shape for mask: " + str(mask.shape))
        elif np.all(mask.shape != self.microlenses_nb):
            raise ValueError("Wrong shape for mask: " + str(mask.shape))
        self.__mask = mask
        if np.all(np.equal(mask, self.__mask)):
            self.__compute_inverse_interference_matrix()
            self.__compute_zonal_matrices()

    @current_image.setter
    def current_image(self, image):
        if type(image) is not np.ndarray:
            image = np.array(image)
        if len(image.shape) != 2:
            raise ValueError("Wrong dimensions for image")
        # elif image.shape[0] != image.shape[1]:
        #     raise ValueError("Wrong shape for image: " + str(image.shape))
        elif np.all(image.shape != self.image_resolution):
            raise ValueError(
                f"Wrong shape for image: {image.shape} should be {self.image_resolution}"
            )

        if (not self.__current_image.shape == image.shape) or (
            not np.all(np.equal(self.__current_image, image))
        ):
            if np.all(self.__microlenses_nb > 1):
                self.__compute_mask
                self.__current_image = image
                self.__compute_splitted_image()
                if self.auto_mask:
                    self.__compute_mask()
            else:
                self.__current_image = image
                self.__splitted_image = image.reshape(1, 1, *image.shape)

    @property
    def splitted_image(self):
        return self.__splitted_image

    @property
    def image_resolution(self):
        return self.__image_resolution

    @property
    def microlenses_nb(self):
        return self.__microlenses_nb

    @property
    def microlenses_focal(self):
        return self.__microlenses_focal

    @property
    def inverse_interference_matrix(self):
        return self.__inverse_interference_matrix

    @property
    def interference_matrix(self):
        return self.__interference_matrix

    def __compute_splitted_image(self):
        split1 = np.array(
            np.split(
                zoom(
                    self.current_image,
                    (
                        (self.current_image.shape[0] // self.microlenses_nb[0])
                        * self.microlenses_nb[0]
                        / self.current_image.shape[0],
                        (self.current_image.shape[1] // self.microlenses_nb[1])
                        * self.microlenses_nb[1]
                        / self.current_image.shape[1],
                    ),
                ),
                self.microlenses_nb[0],
                axis=0,
            )
        )
        splitsplit = np.array(np.split(split1, self.microlenses_nb[1], axis=2))
        self.__splitted_image = np.swapaxes(splitsplit, 0, 1)

    def __compute_mask(self):
        mean_image = self.splitted_image.mean(axis=(3, 2))
        filtered_image = self.splitted_image.mean(axis=(3, 2)) > 0.2 * np.max(
            mean_image
        )
        self.mask = filtered_image

    def __compute_interference_matrix(self):
        final_array = []
        for mode_i in range(2, self.mode_max + 1):
            final_array.append(
                # np.ravel((np.hstack(self.theoritical_slope_mode(mode_i)[:, self.mask])))
                np.ravel((np.hstack(self.theoritical_zernike_mode(mode_i)[self.mask])))
            )

        self.__interference_matrix = np.array(final_array)

    def __compute_zonal_matrices(self):
        mmmm, nnnn = self.mask.shape[:2]
        ds = 1
        self.__zonal_E = np.array(
            np.zeros([(mmmm - 1) * nnnn + (nnnn - 1) * mmmm, nnnn * mmmm])
        )
        for i in range(mmmm):
            for j in range(nnnn - 1):
                self.__zonal_E[i * (nnnn - 1) + j, i * nnnn + j] = -1 / ds
                self.__zonal_E[i * (nnnn - 1) + j, i * nnnn + j + 1] = 1 / ds
        for i in range(nnnn):
            for j in range(mmmm - 1):
                self.__zonal_E[mmmm * (nnnn - 1) + i * (mmmm - 1) + j, i + j * nnnn] = (
                    -1 / ds
                )
                self.__zonal_E[
                    mmmm * (nnnn - 1) + i * (mmmm - 1) + j, i + (j + 1) * nnnn
                ] = (1 / ds)
        self.__zonal_E = np.linalg.pinv(self.__zonal_E)

        self.__zonal_C = np.array(
            np.zeros([(mmmm - 1) * nnnn + (nnnn - 1) * mmmm, 2 * nnnn * mmmm])
        )
        for i in range(mmmm):
            for j in range(nnnn - 1):
                self.__zonal_C[i * (nnnn - 1) + j, i * nnnn + j] = 0.5
                self.__zonal_C[i * (nnnn - 1) + j, i * nnnn + j + 1] = 0.5
        for i in range(nnnn):
            for j in range(mmmm - 1):
                self.__zonal_C[
                    mmmm * (nnnn - 1) + i * (mmmm - 1) + j, nnnn * (mmmm + j) + i
                ] = 0.5
                self.__zonal_C[
                    mmmm * (nnnn - 1) + i * (mmmm - 1) + j, nnnn * (mmmm + j + 1) + i
                ] = 0.5

    def __compute_inverse_interference_matrix(self):
        self.__compute_interference_matrix()
        self.__inverse_interference_matrix = np.linalg.pinv(self.interference_matrix)

    def cog_measurement(self, image):
        self.current_image = image
        slopes = np.empty((2, self.microlenses_nb[0], self.microlenses_nb[1]))
        slopes[:, :, :] = np.nan
        if np.all(self.microlenses_nb > 1):
            sv = self.splitted_image[self.mask]
            slv = slopes[:, self.mask]
            for i in range(sv.shape[0]):
                slv[:, i] = cog(sv[i]).ravel()
            slopes[:, self.mask] = slv - sv[0].shape[0] // 2
        else:
            slopes = (
                cog(self.splitted_image[0, 0, :, :])
                - self.splitted_image.shape[-1] // 2
            )

        return slopes - self.__flat

    def theoritical_zernike_mode(self, mode):
        locweight = np.mean(np.array((self.__n_pixel_subarray / 8,)))
        phase = np.array(
            noll_zernike(
                mode,
                self.__r,
                self.__ang,
                weight=locweight,
                circle=False,
            )
        )
        return phase

    def theoritical_slope_mode(self, mode):
        locweight = np.mean(np.array((self.__n_pixel_subarray / 8,)))
        slope = np.array(
            noll_gradZernike(
                mode,
                self.__r,
                self.__ang,
                weight=locweight,
                circle=False,
            )
        )
        return slope

    def add_flat(self, image):
        self.__flat = self.cog_measurement(image)

    # def reconstruction_cog(self, cog, mode_max=20):
    #     slopes_stack = np.ravel(np.hstack(cog)) * self.microlenses_pitch / 2
    #     return np.nansum(
    #         slopes_stack.reshape(-1, 1) * self.inverse_interference_matrix,
    #         axis=0,
    #     )

    def projection_modal(self, phase, mode_max=20):
        return np.nansum(
            np.nan_to_num(phase[self.mask]).reshape(-1, 1)
            * self.inverse_interference_matrix,
            axis=0,
        )

    def reconstruction_zonal_cog(self, cog):
        phase = np.dot(
            self.__zonal_E,
            np.dot(self.__zonal_C, np.nan_to_num(cog[::-1, :, :].flatten())),
        ).reshape(self.mask.shape[:2])
        phase[~self.mask] = np.nan
        return phase

    def reconstruction_zonal(self, image):
        cog = self.cog_measurement(image)
        cog[:, ~self.mask] = np.nan
        return self.reconstruction_zonal_cog(cog)
        # print(self.zonal_E)
        # return self.reconstruction_cog(
        #     self.cog_measurement(image)[:, self.mask], mode_max=mode_max
        # )

    def reconstruction_modal_cog(self, cog, mode_max=20):
        # pour le moment, seulement zonal+projection"
        return self.projection_modal(
            self.reconstruction_zonal_cog(cog), mode_max=mode_max
        )

    def reconstruction_modal(self, image, mode_max=20):
        # pour le moment, seulement zonal+projection"
        return self.projection_modal(
            self.reconstruction_zonal(image), mode_max=mode_max
        )


class Slope_reconstructor:
    def __init__(self, width_aperture=50, centre_pupille=50, mask=1):

        self.width_aperture = width_aperture
        self.centre_pupille = centre_pupille
        self.__y, self.__x = np.mgrid[
            -1 : 1 : self.width_aperture * 1j, -1 : 1 : self.width_aperture * 1j
        ]
        self.__r = np.sqrt(self.__y**2 + self.__x**2)
        self.mask = mask * (self.__r < 1) * 1
        self.__ang = np.arctan2(self.__y, self.__x)
        self.__inverse_interference_matrix = None
        self.__flat = 0

    def split_image(self, image):
        h_size = image.shape[0] / 2
        split = []
        split.append(
            image[
                np.int(h_size - self.centre_pupille - self.width_aperture / 2) : np.int(
                    h_size - self.centre_pupille + self.width_aperture / 2
                ),
                np.int(h_size + self.centre_pupille - self.width_aperture / 2) : np.int(
                    h_size + self.centre_pupille + self.width_aperture / 2
                ),
            ]
        )
        split.append(
            image[
                np.int(h_size + self.centre_pupille - self.width_aperture / 2) : np.int(
                    h_size + self.centre_pupille + self.width_aperture / 2
                ),
                np.int(h_size + self.centre_pupille - self.width_aperture / 2) : np.int(
                    h_size + self.centre_pupille + self.width_aperture / 2
                ),
            ]
        )
        split.append(
            image[
                np.int(h_size + self.centre_pupille - self.width_aperture / 2) : np.int(
                    h_size + self.centre_pupille + self.width_aperture / 2
                ),
                np.int(h_size - self.centre_pupille - self.width_aperture / 2) : np.int(
                    h_size - self.centre_pupille + self.width_aperture / 2
                ),
            ]
        )
        split.append(
            image[
                np.int(h_size - self.centre_pupille - self.width_aperture / 2) : np.int(
                    h_size - self.centre_pupille + self.width_aperture / 2
                ),
                np.int(h_size - self.centre_pupille - self.width_aperture / 2) : np.int(
                    h_size - self.centre_pupille + self.width_aperture / 2
                ),
            ]
        )
        return np.nan_to_num(split)

    def cog_measurement(self, image):
        image_split = self.split_image(image)
        gx = -(image_split[0] + image_split[1] - (image_split[2] + image_split[3])) / (
            np.sum(image_split, axis=0) + 1e-16
        )
        gy = (image_split[0] + image_split[3] - (image_split[2] + image_split[1])) / (
            np.sum(image_split, axis=0) + 1e-16
        )
        return (np.array((gy, gx)) - self.__flat) * self.mask

    def automatic_mask(self, splitted_image):
        mean_image = splitted_image.mean(axis=(3, 2))
        filtered_image = splitted_image.mean(axis=(3, 2)) > 0.5 * np.max(mean_image)
        return filtered_image

    def theoritical_slope_mode(self, mode):
        return noll_gradZernike(mode, self.__r, self.__ang, weight=5) * (
            self.mask,
            self.mask,
        )

    def interference_matrix(self, mode_max=20):
        final_array = []
        for mode_i in range(2, mode_max + 1):
            final_array.append(
                np.ravel((np.hstack(self.theoritical_slope_mode(mode_i))))
            )
        return np.array(final_array)

    @property
    def inverse_interference_matrix(self):
        if np.all(self.__inverse_interference_matrix) is None:
            self.__inverse_interference_matrix = np.linalg.pinv(
                self.interference_matrix()
            )
        return self.__inverse_interference_matrix

    def add_flat(self, image):
        self.__flat = self.cog_measurement(image)

    def reconstruction(self, image):
        slopes_stack = np.ravel(np.hstack(self.cog_measurement(image)))
        return np.dot(slopes_stack, self.inverse_interference_matrix) * np.array(
            [1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1]
        )


class Irradiance_reconstructor:
    def __init__(self, func, contrast=632.8e-9 / 4, max_noll=12, name="gen"):
        """Camera read should return a float64"""
        self.__contrast = contrast
        self.__max_noll = max_noll
        self.__full_name = "calibration_data_" + name
        self.__func = func
        self.__flat = 0
        try:
            self.calibration_data = np.load("/tmp/" + self.__full_name + ".npy")
        except FileNotFoundError:
            warnings.warn(
                "No calibration data, will need to run full calibration sequence (slower)",
                Warning,
            )
            self.calibration_data = None

    @property
    def max_noll(self):
        return self.__max_noll

    @property
    def contrast(self):
        return self.__contrast

    def calibration_sequence(self, path="/tmp/"):
        self.interaction_matrix = self.construct_interaction_matrix()
        self.save_interaction_matrix(path)
        self.inference_matrix = self.construct_inference_matrix()
        self.inverse_inference_matrix = np.linalg.pinv(self.inference_matrix)

    def flush_calibration(self):
        self.calibration_data = None

    def normalise_irradiance(self, irradiance):
        return np.nan_to_num((irradiance / np.sum(np.sum(irradiance))) - self.__flat)

    def add_flat(self, image):
        self.__flat = self.normalise_irradiance(image)

    def construct_interaction_matrix(self):
        """Get the response of the system for each aberration using
        simulations to get the pattern."""
        if self.calibration_data is not None:
            return self.calibration_data
        else:
            inter_matrix = []
            for noll_index in range(2, self.max_noll):
                contrP = self.normalise_irradiance(
                    self.__func(noll_index, self.contrast)
                )
                contrM = self.normalise_irradiance(
                    self.__func(noll_index, -2 * self.contrast)
                )
                self.__func(noll_index, self.contrast)
                inter_matrix.append(np.array([(contrP - contrM) / 2.0])[0])
                self.__func(1, 0)
            return np.array(inter_matrix)

    def construct_inference_matrix(self):
        return np.array(
            [
                np.ravel(self.interaction_matrix[x])
                for x in range(self.interaction_matrix.shape[0])
            ]
        ).T

    def reconstruction(self, irradiance):
        p_abb = np.ravel(self.normalise_irradiance(irradiance))
        p_inv = self.inverse_inference_matrix
        decomp_abb = np.dot(p_inv, p_abb)
        return decomp_abb * self.contrast

    def save_interaction_matrix(self, path):
        np.save(path + "/" + self.__full_name, self.interaction_matrix)
