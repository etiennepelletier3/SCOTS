from lri_ao.utils import cog
import numpy as np
import unittest


class TestCog(unittest.TestCase):
    def test_cog(self):
        for x in range(20):
            for res in (251, 250):
                cog_test = (
                    np.random.random(
                        (
                            2,
                            1,
                            1,
                        )
                    )
                    - 0.5
                )
                cog_test *= res / 2
                cog_test += res / 2

                cog_test = cog_test.astype("int")

                cog_input = np.copy(cog_test)
                cog_input += np.array(
                    np.mgrid[
                        slice(0, 1, 1),
                        slice(0, 1, 1),
                    ]
                    * res
                )
                image_test = np.zeros((res * 2, res))
                image_test[cog_input[0], cog_input[1]] = 5
                image_test += np.int(np.random.rand(1)[0] * 5)

                cog_result = cog(image_test)
                self.assertTrue(
                    np.all(
                        np.isclose(
                            cog_result[0],
                            cog_test[0],
                        )
                    )
                )

    def test_cog_empty(self):
        img = np.zeros((100, 100))
        result = cog(img)
        self.assertTrue(np.all(result == 0))

    def test_cog_speed(self):
        y, x = np.mgrid[-1 : 1 : 2456 * 1j, -1 : 1 : 2054 * 1j]
        func = 2 * np.exp(-1000 * ((x - 0.5) ** 2 + (y + 0.5) ** 2))
        func += np.random.rand(*func.shape)
        # import matplotlib.pyplot as plt

        # plt.imshow(cog(func+10))
        # plt.show()
        # plt.imshow(cog(func))
        # plt.show()
        # print(cog(func))
        from time import process_time

        tick = process_time()
        # print(cog(func+10))
        print(cog(func[::4, ::4]) * 4)
        # print(cog(func))
        print(1 / (process_time() - tick))

    # def test_cog_image(self):
    #     img = np.array(
    #         Image.open(
    #             Path(__file__)
    #             .parent.resolve()
    #             .with_stem("media")
    #             .joinpath("image_centroide.bmp")
    #         )
    #     )
    #     img2 = img


if __name__ == "__main__":
    # change current file path file name to "xalut.py"

    # print(Path(__file__).parent)
    # Image.open(__files"")
    unittest.main()
