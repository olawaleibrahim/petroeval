from math import log2
from typing import Dict, List


class GrainPreprocess:

    def __init__(
        self,
        sieve_size_in_mm: List[float] = [
            2,
            1.18,
            0.85,
            0.425,
            0.300,
            0.150,
            0.075,
            0.063,
        ],
    ):
        """Intialization of the GrainPreprocess Class.

        Args:
            sieve_size_in_mm (List[float], optional): _description_. Defaults to [ 2, 1.18, 0.85, 0.425, 0.300, 0.150, 0.075, 0.063, ].

        sieve_size_in_mm is default to the seive size on a stack/seive shaker used in the analysis of grains of sandstone.
        """
        self.stack_sieve_sizes_in_mm = sieve_size_in_mm

    def stack_to_phi_scale(self):
        """Stack_to_Phi_Scale

        This function converts the sieve sizes in mm to phi scale.
        Here is a resource to cross_check mm to phi_scale conversation : https://www.youtube.com/watch?v=AxjC-KiIuC8
        Returns:
            List[float]: Stack_Of_Sieve_Sizes_in_Phi_Scale
        """

        stack_of_sieves_in_phi_scale = [
            round(-(log2(sieve_size) / log2(2)), 2)
            for sieve_size in self.stack_sieve_sizes_in_mm
        ]
        return stack_of_sieves_in_phi_scale

    def bed_thickness_in_percentage(self, measured_bed_thickness: List[float]):
        """Bed_Thickness_In_Percentage

        This method converts the measured bed thickess of the field to a scale with 0-100
        Returns:
            list[float]: Array of Bed_thickness between 0 and 100
        """

        bounded_bed_thickness: List[float] = [
            (round(bed / (sum(measured_bed_thickness)) * 100), 2)
            for bed in measured_bed_thickness
        ]

        return bounded_bed_thickness

    def bed_thickness_accumlation(self, bounded_bed_thickness: list[float]):
        """Bed Thickness Acumulation

            This method calculates the cumulative sum of the thickness within the bed.
        Args:
            bounded_bed_thickness (list[float]): There is a constraint that prevents the sum of the bounded_bed_thickness to be greater 100

        Raises:
            ValueError: Raise this error when the sum of the bounded_bed_thickness is greater than 100

        Returns:
            List[float]: An array of the cumulative sum of the thickness of a bed
        """

        sum_validation = sum(bounded_bed_thickness)

        if sum_validation > 100:
            raise ValueError(
                f"The total thickness of the evaluated bed in greater than 100: it is {sum_validation}"
            )

        accumulated_bed_thickness: List[float] = []
        increment_by = 0
        for bed_thickness in bounded_bed_thickness:
            increment_by += bed_thickness
            accumulated_bed_thickness.append(round(increment_by, 2))

        return accumulated_bed_thickness

    def bed_percentiles(
        self, accumlated_bed_thickness: List[float]
    ) -> Dict[str, float]:
        """Bed Percentiles
        This method returns a dictionary of the standard sedimentological passings and the accumulated_sum of bed_thickness.
        The Sedimentology Passing 5%,25%, 16%, 50%, 75%, 84%, 95%, 100%
        Args:
            accumlated_bed_thickness (List[float]): This are the accumlated sum of the bed thickness.

        Returns:
            Dict[str, float]: This is the passing with the coresponding thickness
        """

        SED_PASSING: List[str] = [
            "5%",
            "16%",
            "25%",
            "50%",
            "75%",
            "84%",
            "95%",
            "100%",
        ]
        return dict(zip(SED_PASSING, accumlated_bed_thickness))
