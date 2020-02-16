import numpy as np
import unittest

from sapsan.utils.shapes import split_cube_by_grid, combine_cubes


def generate_test_cube():
    return np.moveaxis(
        np.vstack([
            np.vstack([
                np.hstack([
                    np.vstack([
                        np.ones((16, 16, 3))*0,
                        np.ones((16, 16, 3))*64
                    ]),
                    np.vstack([
                        np.ones((16, 16, 3))*128,
                        np.ones((16, 16, 3))*192
                    ])
                ]).reshape(-1, 32, 32, 3) for _ in range(16)
            ]),
            np.vstack([
                np.hstack([
                    np.vstack([
                        np.ones((16, 16, 3))*256,
                        np.ones((16, 16, 3))*320
                    ]),
                    np.vstack([
                        np.ones((16, 16, 3))*384,
                        np.ones((16, 16, 3))*448
                    ])
                ]).reshape(-1, 32, 32, 3) for _ in range(16)
            ])
        ]),
        -1, 0
    )


class TestDatasetUtils(unittest.TestCase):
    """ Dataset utils test. """

    def setUp(self) -> None:
        self.cube = generate_test_cube()

    def test_cube_split(self):
        """ Test correctness of checkpoint cube split. """
        batched = split_cube_by_grid(self.cube, 32, 16, 3)
        for subcube in batched:
            subcube = subcube.reshape(-1)
            value = subcube[0]
            self.assertTrue(np.all(subcube == value))

    def test_cube_split_and_combine(self):
        """ Test correctness of split and restore. """
        batched = split_cube_by_grid(self.cube, 32, 16, 3)
        restored_cube = combine_cubes(batched, 32, 16)
        self.assertTrue(np.all(restored_cube == self.cube))


