## Formation Evaluation Package

Formation evaluation module for geoscientists, petrophysicists for easy and quick quantitative petrophysical analysis.

Features include:

-   Calculating petrophysical parameters for reservoir quantization:
        Parameters like Total Porosity, Effective Porosity, Water Saturation, Oil Saturation

-   Estimating Reservoir Volumes:
    - Gross rock Volume
    - Net to Gross Volume
    - Net Pay of Reservoir section

-   Predicting well lithofacies and lithology values using machine learning

-   Data preparation and preprocessing for machine learning us use

-   Visualizing well logs

-   Reading las files

-   Resolving missing values in well logs

-   Automatic correction of estimated properties

### QUICK TUTORIAL

    ## reading a single lasio file 
    from petroeval import read_lasio
    las = read_lasio('WLC_PETRO_COMPUTED_INPUT_1.LAS')
    df = las.df()

    ## reading multiple lasio files
    from petroval import read_lasios
    las1='WLC_PETRO_COMPUTED_INPUT_1.LAS'
    las2='WLC_PETRO_COMPUTED_INPUT_1.LAS'
    las=read_lasios(las1,las2) =>returns a list of the read lasio objects<br>

    ## import modules and functions
    import numpy
    from petroeval import evaluate_reservoir
    from petroeval.visualizations import summary, log_plot

    # Create an instance of the reservoir section passing in required arguments
    from evaluate_reservoir import FormationEvaluation
    ...

Tutorial repo link:     
https://github.com/olawaleibrahim/beta-tests/tree/main/petroeval
