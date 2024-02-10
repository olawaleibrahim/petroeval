from math import log2
from typing import Dict, List


class GrainPreprocess:
    """Grain Process class"""

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

    def bed_sieve_reminant_in_percentage(self, measured_seive_reminant: List[float]):
        """Bed_Thickness_In_Percentage

        This method converts the measured bed thickess of the field to a scale with 0-100
        Returns:
            list[float]: Array of Bed_thickness between 0 and 100
        """

        bounded_bed_sieve_remenant: List[float] = [
            (round(seive_reminant / (sum(measured_seive_reminant)) * 100), 2)
            for seive_reminant in measured_seive_reminant
        ]

        return bounded_bed_sieve_remenant

    def bed_reminant_accumlation(self, bounded_bed_sieve_reminant: list[float]):
        """Bed Reminant Acumulation

            This method calculates the cumulative sum of the reminant within the bed.
        Args:
            bounded_bed_sieve_reminant (list[float]): There is a constraint that prevents the sum of the bounded_bed_reminant to be greater 100

        Raises:
            ValueError: Raise this error when the sum of the bounded_bed_sieve_reminant is greater than 100

        Returns:
            List[float]: An array of the cumulative sum of the thickness of a bed
        """

        sum_validation = sum(bounded_bed_sieve_reminant)

        if sum_validation > 100:
            raise ValueError(
                f"The total thickness of the evaluated bed in greater than 100: it is {sum_validation}"
            )

        accumulated_bed_reminant: List[float] = []
        increment_by = 0
        for bed_sieve_reminant in bounded_bed_sieve_reminant:
            increment_by += bed_sieve_reminant
            accumulated_bed_reminant.append(round(increment_by, 2))

        return accumulated_bed_reminant

    def bed_percentiles(self, accumlated_bed_reminant: List[float]) -> Dict[str, float]:
        """Bed Percentiles
        This method returns a dictionary of the standard sedimentological passings and the accumulated_sum of bed_thickness.
        The Sedimentology Passing 5%,25%, 16%, 50%, 75%, 84%, 95%, 100%
        Args:
            accumlated_bed_reminant (List[float]): This are the accumlated sum of the bed thickness.

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
        return dict(zip(SED_PASSING, accumlated_bed_reminant))
