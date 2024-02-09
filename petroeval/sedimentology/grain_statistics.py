from typing import List
from pandas import DataFrame


class grain_statistics:

    def __init__(self, grain_sizes_in_phi: List[float]) -> None:

        self.grain_sizes_in_phi = self.__grain_size_in_phi(
            grain_size_in_phi=grain_sizes_in_phi
        )

    def __grain_size_in_phi(self, grain_size_in_phi: List[float]):
        """Grain Size in Phi

        Args:
            grain_size_in_phi (List[float]): [Φ5,Φ16, Φ25, Φ50, Φ75,Φ84,Φ95]
        Raises:
            ValueError: The value is less than or greater than the len of the expected data

        Returns:
            List[float]: [Φ5,Φ16, Φ25, Φ50, Φ75,Φ84,Φ95]
        """

        if len(grain_size_in_phi) < 7 or len(grain_size_in_phi) > 7:
            raise ValueError(
                "The grain size is does not conform to the expected phi_sizes, here are the sizes :[Φ5,Φ16, Φ25, Φ50, Φ75,Φ84,Φ95]"
            )
        return grain_size_in_phi

        ...

    def graphic_mean(self):
        """Graphic Mean

        formular: MZ = (Φ I6 + Φ5O + Φ 84)/3
        Returns:
            float: MZ
        """
        grain_sizes = self.grain_sizes_in_phi

        return round((grain_sizes[1] + grain_sizes[3] + grain_sizes[5]) / 3, 2)
        ...

    def graphic_standard_deviation(self):
        """Graphic Standard Deviation

        formular: ϬI= (Φ84 - Φ16)/4 + (Φ95- Φ5)/6.6

        Returns:
            float: ϬI
        """

        grain_sizes = self.grain_sizes_in_phi
        return round(
            (grain_sizes[5] + grain_sizes[1]) / 4
            + (grain_sizes[6] - grain_sizes[0]) / 6.6,
            2,
        )

    def graphic_standard_kurtosis(self):
        """Graphic Standard Kurtosis

            formular: KG = (Φ95 - Φ5) / [2.44 (Φ75 - Φ25)]

        Returns:
            float: KG
        """

        grain_sizes = self.grain_sizes_in_phi

        return round(
            (grain_sizes[6] - grain_sizes[0])
            / (2.44 * (grain_sizes[4] - grain_sizes[2])),
            2,
        )

        ...

    def graphic_skewness(self):
        """Graphic Skewness


            formular: Sk=(Φ16+ Φ84-2Φ50) / 2(Φ84- Φ16)] + [(Φ5+ Φ95-2 Φ50) / 2(Φ95- Φ5)


        Returns:
            float: Sk
        """

        grain_sizes = self.grain_sizes_in_phi

        left_formular = round(
            grain_sizes[1]
            + grain_sizes[5]
            - (2 * grain_sizes[3]) / (2 * (grain_sizes[5] - grain_sizes[1])),
            2,
        )
        right_formular = round(
            grain_sizes[0]
            + grain_sizes[6]
            - (2 * grain_sizes[3]) / 2 * (grain_sizes[6] - grain_sizes[0]),
            2,
        )

        return round(left_formular + right_formular, 2)

    def data_to_dataframe(
        self,
        name_of_bed: str,
        graphic_mean: float,
        graphic_standard_dev: float,
        graphic_skewness: float,
        graphic_kurtosis: float,
    ):
        """Data_to_Dataframe

        This method create a Dataframe of the graphic mean, standard deviation, skewness and kurtosis
        Args:
            name_of_bed (str):eg Bed A, Bed Alpha
            graphic_mean (float):MZ
            graphic_standard_dev (float): ϬI
            graphic_skewness (float): Sk
            graphic_kurtosis (float): KG

        Returns:
            DataFrame
        """

        bed_stat_data = DataFrame(
            {
                "Name_of_bed": [name_of_bed],
                "Graphic Mean": [graphic_mean],
                "Graphic Standard deviation": [graphic_standard_dev],
                "Graphic Skewness": [graphic_skewness],
                "Graphic Kurtosis": [graphic_kurtosis],
            }
        )

        return bed_stat_data

    def to_csv(self, data_frame: DataFrame, name_of_rock_unit: str):
        """
        Saves Bed Grains stats from the CompleteBeds and creates a CSV file in your Present working Directory.
        """

        data_frame.to_csv(f"{name_of_rock_unit}.csv")
