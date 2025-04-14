# EcotoxModelFitting

## Quickstart


## Configuring a fitting problem 

A fitting problem is defined by initializing a `ModelFit` structure, 
and filling in the arguments needed to fully specify the problem. <br>
Once that is done, there are several methods available to conduct the actual calibration 
(e.g. Population Monte Carlo, local optimization, global optimization).

### Data weights

`EcotoxModelFitting.jl` allows you to assign weights to data on two levels: 

- On the level of the response variable, through the argument `data_weights` in the `ModelFit` constructor
- On the level of an individual variable, by adding a column `observation_weight` to the raw data.

The `data_weights` and `observation_weight`s will be normalized individually during construction of the `ModelFit` object. <br>
Changing either one requires to re-call `generate_loss_function`.

## Guidelines for data organization

Within `EcotoxModelFitting.jl`, a dataset is an `OrderedDict` of `DataFrame`s. <br>


```Julia
using DataStructures

data = OrderedDict(
    :growth => df_growth,
    :repro => df_repro,
    :survival => df_survival,
    :exposure => df_exposure
)

```
`data` is a dataset and `data[:growth]`, `data[:repro]`, etc. are *data keys*. <br>

### Data storage on disc

On disc, it is best to store all files which make up a dataset in a dedicated subdirectory, e.g.

```bash
myproject # project directory
    data # sub-directory containing all data related to the project
        exp_raw # sub-directory containing raw experimental data
            experiment1
                growth.csv # measured growth over time
                repro.csv # measured reproduction over time
                survival.csv # observed survival over time
                exposure.csv # exposure scenarios
                meta.yml # metadata
            experiment2
                growth.csv
                repro.csv
                survival.csv
                exposure.csv
                meta.yml
```

We recommend to stick to this organizational format, even if a dataset contains only of a single table. <br>
This format is also compatible with data management systems like [datalad](https://www.datalad.org/). <br>
Note that in the example above, each dataset also contains a `meta.yml` file. <br>
This file contains all the metadata pertinent to the dataset. 
An example is given in the examples subdirectory on Github. <br>


### Tidy data frames
We assume each `DataFrame` to be organized in tidy format *sensu* Wickham, see the [corresponding publication](https://www.jstatsoft.org/article/view/v059i10/). <br><br>

In short, this means that data tables are *column-oriented*. <br>
Each column represents a variable (e.g. time, treatment, organism length,...). <br>
Each row represent an observation. For example, organism length $L$ at time-point $t$ in treatment $T$. <br>
If you select a row of your table and it contains measurements at multiple time-points, 
or for multiple treatments, the data is not tidy. <br><br>
This explicitly excludes formats in which different treatments are represented by different columns. <br>
If you are not sure whether your data is tidy, check whether you can apply common data queries, e.g.

```Julia
using DataFramesMeta 

df_end = @subset(df, :t_day .== (maximum(:t_day )), :treatment .== "control")

```

to select the final time-point of the control in the dataframe `df`, assuming that `df` has a column `t_day` indicating time and `treatment` encoding the treatment. <br><br>



