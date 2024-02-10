from unittest import TestCase, main
from petroeval.sedimentology.grain_statistics import GrainStatistics


class TestGrainStatistics(TestCase):

    def _grain_stats(self):
        return GrainStatistics(
            grain_sizes_in_phi=[-0.65, 0.15, 0.65, 1.75, 2.45, 2.65, 3.45]
        )

    def test_graphic_mean(self):

        graphic_mean = self._grain_stats().graphic_mean()
        self.assertEquals(first=graphic_mean, second=1.52)

    def test_graphic_standard_deviation(self):
        graphic_standard_dev = self._grain_stats().graphic_standard_deviation()
        self.assertEquals(
            first=graphic_standard_dev,
            second=1.25,
        )

    def test_graphic_standard_kurtosis(self):

        graphic_standard_kurtosis = self._grain_stats().graphic_standard_kurtosis()

        self.assertEquals(first=graphic_standard_kurtosis, second=3.02)

    def test_graphic_skewness(self):
        graphic_skewness = self._grain_stats().graphic_skewness()

        self.assertEquals(first=graphic_skewness, second=-2.31)


if __name__ == "__main__":
    main()
