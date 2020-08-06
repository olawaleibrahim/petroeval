## Formation Evaluation Package

Formation evaluation module for geoscientists, petrophysicists for easy and quick quantitative petrophysical analysis.

Features include:

-   Calculating petrophysical parameters for reservoir quantization:
        Parameters like Total Porosity, Effective Porosity, Water Saturation, Oil Saturation

-   Estimating Reservoir Volumes:
                Gross rock Volume
                Net to Gross Volume
                Net Pay of Reservoir section

-   Visualizing well logs for correlation

-   Resolving missing values in well logs

-   Automatic correction of estimated properties

Subsequent release features:

-   Total oil or gas in place
-   Permeability of reservoir section
-   Official documentations to follow

### QUICK TUTORIAL
'''python
         ## reading  a single lasio file <br>
         from petroeval import read_lasio<br>
         las = read_lasio('WLC_PETRO_COMPUTED_INPUT_1.LAS')<br>
         df = las.df()<br>

         #reading  multiple lasio files<br>

         from petroval import read_lasios<br>
         las1='WLC_PETRO_COMPUTED_INPUT_1.LAS'<br>
         las2='WLC_PETRO_COMPUTED_INPUT_1.LAS'<br>
         las=read_lasios(las1,las2) =>returns a list of the read lasio objects<br><br>

        # import modules and functions<br>

        import numpy<br>
        from petroeval import evaluate_reservoir<br>
        from petroeval.visualizations import summary, log_plot<br>

        # Create an instance of the reservoir section passing in required arguments<br>

        from evaluate_reservoir import FormationEvaluation<br
 '''
Tutorial link: https://github.com/olawaleibrahim/petroeval/blob/master/petroeval/Tutorial.ipynb
