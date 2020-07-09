# Formation Evaluation Package

## Formation evaluation module for geoscientists, petrophysicists for easy and quick quantitative petrophysical analysis.

## Features include:

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

### QUICK TUTORIAL

### import file (csv, lasio)
`import lasio` <br>
`las = lasio.read('WLC_PETRO_COMPUTED_INPUT_1.LAS')` <br>
`df = las.df()`
### import modules and functions
`import numpy` <br>
`import evaluate_reservoir` <br>
`from visualizations import summary, log_plot`<br>

## create an instance of the reservoir section passing in required arguments
- `from evaluate_reservoir import FormationEvaluation`

## Tutorial can be found on 
- `https://github.com/olawaleibrahim/petroeval/blob/master/Tutorial.ipynb`
