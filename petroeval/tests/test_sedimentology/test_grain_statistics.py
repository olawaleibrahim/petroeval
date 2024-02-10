from unittest import TestCase, main
from petroeval.sedimentology.grain_preprocess import GrainPreprocess


class TestGrainPreprocess(TestCase):

    def test_stack_to_phi_scale(self):
        grain_preprocess = GrainPreprocess()

        grain_size_in_phi_scale = grain_preprocess.stack_to_phi_scale()

        self.assertEquals(
            first=grain_size_in_phi_scale,
            second=[-1.0, -0.24, 0.23, 1.23, 1.74, 2.74, 3.74, 3.99],
        )


if __name__ == "__main__":
    main()
