from multiprocessing import Pool, cpu_count, freeze_support

import numpy as np
import pandas as pd

import silicone.time_projectors as tp
import silicone.utils

"""
This script measures how accurate the different projectors are at recreating known data.
We remove the data after 2050 and infill to find it and compare the true and infilled 
values. It normalises this difference by the total range of the data. It runs on 
multiple cores and saves the resulting statistics to disk. It may run for either a list
of specified cases, or for all possible cases. Optionally it can also 
plot graphs comparing the infilled and original data - this is best done with only a 
small number of cases, otherwise a huge number of files will be created.  

The options controlling how it works are listed below.
"""


def main():
    freeze_support()
    # __________________________________Input options___________________________________
    # Where is the file stored for data used to fill in the sheet?
    input_data = "./sr_15_complete.csv"
    # A list of all projectors to investigate, here a reference to the actual projector
    extenders_list = [
        tp.ExtendLatestTimeQuantile
    ]
    options_list = [{}]
    # This list must agree with the above list, but is the name of the projectors
    projectors_name_list = [
        x.__name__ for x in extenders_list
    ]
    # Leader is a single data class presented as a list.
    leaders = ["Emissions|CO2"]
    # Place to save the infilled data as a csv
    save_file = "../Output/projectorResults/projectorComparisonLead_stdv_{}.csv".format(
        leaders[0].split("|")[-1]
    )
    # Do we want to save plots? If not, leave as None, else the location to save them.
    # Note that these are not filter-dependent and so only the results of the last
    # filter will persist
    save_plots = None  #  "../output/projectorResults/plots/"
    # Do we want to run this for all possible filters? If so, choose none,
    # otherwise specify the filter here as a list of tuples
    to_compare_filter = None
    """[
        ("AIM/CGE 2.0", "SSP1-19"),
        ("AIM/CGE 2.1", "TERL_15D_NoTransportPolicy"),
    ]"""
    years = range(2020, 2101, 10)
    test_year = 2050
    variables_investigated = ["Emissions|CO2"]
    # Uncomment the below
    # __________________________________end options_____________________________________
    years_to_investigate = [year for year in years if year > test_year]
    assert len(extenders_list) == len(projectors_name_list)
    assert len(options_list) == len(projectors_name_list)

    db_all = silicone.utils.download_or_load_sr15(
        input_data, valid_model_ids="*"
    ).filter(region="World", year=years)
    db_all.filter(variable=variables_investigated, inplace=True)
    # This is the model/scenario combination to compare.
    if to_compare_filter:
        all_possible_filters = to_compare_filter
    else:
        all_possible_filters = (
            db_all.data[["model", "scenario"]]
            .groupby(["model", "scenario"])
            .size()
            .index.values
        )

    all_args = [
        (
            filter_instance,
            db_all,
            variables_investigated,
            projectors_name_list,
            extenders_list,
            save_plots,
            leaders,
            years_to_investigate,
            options_list,
        )
        for filter_instance in all_possible_filters
    ]

    # Perform the loop
    with Pool(cpu_count() - 1) as pool:
        results_db = list(pool.map(_recalc_and_compare_results, all_args))

    results_count = sum([result.notnull() for result in list(results_db)])
    overall_results = sum([result.fillna(0) for result in list(results_db)])
    overall_results = overall_results / results_count
    overall_results.to_csv(save_file)


def _recalc_and_compare_results(args):
    (
        one_filter,
        db_all,
        vars_to_crunch,
        projectors_name_list,
        projectors_list,
        save_plots,
        leaders,
        years_to_investigate,
        options_list,
    ) = args
    combo_filter = {"model": one_filter[0], "scenario": one_filter[1]}
    investigated_scen_df = db_all.filter(**combo_filter)
    input_to_fill = investigated_scen_df.filter(
        year=years_to_investigate, keep=False
    )
    results_db = pd.DataFrame(index=vars_to_crunch, columns=projectors_name_list)
    if leaders not in input_to_fill.variables(False).values:
        print(
            "No data for {} in model {}, scen {}".format(
                leaders, one_filter[0], one_filter[1]
            )
        )
        return results_db
    # Remove all items that overlap directly with this
    db_filter = db_all.filter(**combo_filter, keep=False)
    # Set up normalisation
    norm_factor = pd.Series(index=years_to_investigate, dtype=float)

    for projector_ind in range(len(projectors_list)):
        for var_inst in vars_to_crunch:
            for year in norm_factor.index:
                norm_factor[year] = np.std(
                    db_all.filter(year=year, variable=var_inst).data["value"]
                )
            if norm_factor[year] == 0:
                continue
            originals = investigated_scen_df.filter(
                variable=var_inst, year=years_to_investigate
            ).data.set_index("year")["value"]
            if originals.empty:
                print("No data available for {}".format(var_inst))
                continue
            valid_scenarios = db_filter.filter(variable=var_inst).scenarios()
            db = db_filter.filter(scenario=valid_scenarios)
            # Initialise the object that holds the results
            projector_instance = projectors_list[projector_ind](db)
            filler = projector_instance.derive_relationship(
                var_inst, **options_list[projector_ind]
            )
            interpolated = filler(input_to_fill)
            interp_values = interpolated.data.set_index("year")["value"]
            assert (
                interpolated["year"].size == interpolated["year"].unique().size
            ), "The wrong number of years have returned values"
            # Calculate the RMS difference, normalised by the spread of values
            results_db[projectors_name_list[projector_ind]][var_inst] = (
                np.nanmean(
                    ((interp_values - originals) / norm_factor) ** 2
                )
            ) ** 0.5
        print("Completed projector {}".format(projectors_name_list[projector_ind]))
    return results_db


if __name__ == "__main__":
    main()
