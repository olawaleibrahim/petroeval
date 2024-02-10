from unittest import TestCase, main
from petroeval.sedimentology.grain_preprocess import GrainPreprocess


class TestGrainPreprocess(TestCase):

    def _grain_preprocess(self):
        return GrainPreprocess()

    def test_stack_to_phi_scale(self):

        grain_size_in_phi_scale = self._grain_preprocess().stack_to_phi_scale()

        self.assertEquals(
            first=grain_size_in_phi_scale,
            second=[-1.0, -0.24, 0.23, 1.23, 1.74, 2.74, 3.74, 3.99],
        )

    def test_bed_sieve_reminant_in_percentage(self):
        bed_A = [2.6, 12.4, 17.4, 41.1, 9.9, 10, 4.2, 1]
        bed_sieve_reminant_in_percentage = (
            self._grain_preprocess().bed_sieve_reminant_in_percentage(
                measured_seive_reminant=bed_A
            )
        )
        self.assertEquals(
            first=bed_sieve_reminant_in_percentage,
            second=[2.64, 12.58, 17.65, 41.68, 10.04, 10.14, 4.26, 1.01],
        )

    def test_bed_reminant_accumlation(self):

        bounded_bed_sieve_reminant = [
            2.64,
            12.58,
            17.65,
            41.68,
            10.04,
            10.14,
            4.26,
            1.01,
        ]
        accumulated_bed_reminant = self._grain_preprocess().bed_reminant_accumlation(
            bounded_bed_sieve_reminant=bounded_bed_sieve_reminant
        )
        self.assertEquals(
            first=accumulated_bed_reminant,
            second=[2.64, 15.22, 32.87, 74.55, 84.59, 94.73, 98.99, 100.0],
        )

    def test_bed_percentiles(self):
        accumulated_bed_sieve_reminant = [
            2.64,
            15.22,
            32.87,
            74.55,
            84.59,
            94.73,
            98.99,
            100.0,
        ]
        bed_percentiles = self._grain_preprocess().bed_percentiles(
            accumlated_bed_reminant=accumulated_bed_sieve_reminant
        )
        self.assertEquals(
            first=bed_percentiles,
            second={
                "5%": 1.62,
                "16%": 8.01,
                "25%": 19.67,
                "50%": 59.63,
                "75%": 81.23,
                "84%": 95.94,
                "95%": 99.39,
                "100%": 100.0,
            },
        )


if __name__ == "__main__":
    main()
