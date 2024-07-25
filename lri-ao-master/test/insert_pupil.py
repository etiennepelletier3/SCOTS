import unittest

import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.joinpath(Path(__file__).absolute().parents[1], "src")))

from lri_ao.fields import Pupil


class TestInsertPupil(unittest.TestCase):
    config_a = {}
    config_a["wavelengths"] = [500e-9, 632.8e-9]
    config_a["sampling_points"] = 256
    config_a["side_length"] = 50
    config_a["obscuration_percentage"] = 0.25
    config_a["aperture_diameter"] = 20

    config_b = {}
    config_b["wavelengths"] = 500e-9
    config_b["sampling_points"] = 255
    config_b["side_length"] = 50
    config_b["obscuration_percentage"] = 0.25
    config_b["aperture_diameter"] = 20

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pupil_a = Pupil(**self.config_a)
        self.pupil_b = Pupil(**self.config_b)

    def test_1(self):
        target_size = self.pupil_a.get_padded_phase_size()
        print(target_size)
        target_phase = np.random.random(target_size)
        self.pupil_a.set_padded_phase(target_phase)
        self.pupil_a.display_phase()
        plt.show()

        target_size = self.pupil_b.get_padded_phase_size()
        print(target_size)
        target_phase = np.random.random(target_size)
        target_phase[0] += 3
        self.pupil_b.set_padded_phase(target_phase)
        self.pupil_b.display_phase()
        plt.show()


if __name__ == "__main__":
    unittest.main()
