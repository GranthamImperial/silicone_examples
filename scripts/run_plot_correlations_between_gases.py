from silicone.utils import download_or_load_sr15

import silicone.plotting

# This script shows how to use the investigatory script, _plot_emission_correlations_
# quantile_rolling_windows. It saves plots of the relationship between carbon dioxide
# and other emissions to the '../output' folder
# ________________________________________________________
# We must indicate the main gas of interest
x_gases = [
    "Emissions|PFC",
    "Emissions|CH4",
    "Emissions|CO2|AFOLU",
    "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|CO2",
]
years_of_interest = [2030, 2050, 2100]
save_results = "../output/LotsOfPlotsOfRelations"
# if non-null, also plot these quantiles.
plot_quantiles = None  # [0.05, 0.33, 0.5, 0.67, 0.95]
# if non-null, save data on the quantiles too
quantiles_savename = None
# How many boxes are used to fit the quantiles?
quantile_boxes = 20
# Should we extend the quantile boxes by an additional factor?
quantile_decay_factor = 1
# Color different models different colours?
model_colours = True
# In the model-coloured version, how much does the figure need to be reduced by to leave
# room for the legend?
legend_fraction = 0.65
# ________________________________________________________
# where do we get the data from?
SR15_SCENARIOS = "./sr15_scenarios_more_regions.csv"
sr15_data = download_or_load_sr15(SR15_SCENARIOS)

for ind in range(len(x_gases)):
    x_gas = x_gases[ind]
    silicone.plotting._plot_emission_correlations_quantile_rolling_windows(
        sr15_data,
        save_results,
        years_of_interest,
        x_gas,
        plot_quantiles,
        quantile_boxes,
        quantile_decay_factor,
        model_colours,
        legend_fraction,
    )
