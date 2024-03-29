from multiprocessing import Pool, cpu_count, freeze_support

import numpy as np
import pandas as pd
import pyam

import silicone.multiple_infillers as mi
from silicone.utils import return_cases_which_consistently_split

"""
This script measures how accurate the different ways of using multiple infillers are at 
recreating known data. We remove the data of interest, infill to find it and compare the 
true and infilled values. It normalises this difference by the total range of the data. 
It runs on multiple cores and saves the resulting statistics to disk. It may run for 
either a list of specified cases, or for all possible cases. Optionally it can also 
plot graphs comparing the infilled and original data - this is best done with only a 
small number of cases, otherwise a huge number of files will be created.  

The options controlling how it works are listed below.
"""


def main():
    freeze_support()
    # __________________________________Input options___________________________________
    # Where is the file stored for data used to fill in the sheet?
    input_data = "./sr_15_complete.csv"
    # A list of all crunchers to investigate, here a reference to the actual cruncher
    crunchers_list = [
        mi.SplitCollectionWithRemainderEmissions,
        mi.SplitCollectionWithRemainderEmissions,
    ]
    options_list = [
        {
            "aggregate": "Emissions|CO2",
            "components": ["Emissions|CO2|AFOLU"],
            "remainder": "Emissions|CO2|Energy and Industrial Processes",
        },
        {
            "aggregate": "Emissions|CO2",
            "components": ["Emissions|CO2|Energy and Industrial Processes"],
            "remainder": "Emissions|CO2|AFOLU",
        },
    ]
    # This list must agree with the above list, but is the name of the crunchers
    crunchers_name_list = [
        "Remainder E&I", "Remainder AFOLU"
    ]
    # Leader is a single data class presented as a list.
    leaders = ["Emissions|CO2"]
    # Place to save the infilled data as a csv
    save_file = "../Output/MultipleInfillerResults/MIComparisonLead_stdv_{}.csv".format(
        leaders[0].split("|")[-1]
    )
    save_workings = "../Output/MultipleInfillerResults/MIComparisonLead_stdv_workings_{}.csv".format(
        leaders[0].split("|")[-1]
    )
    # Do we want to save plots? If not, leave as None, else the location to save them.
    # Note that these are not filter-dependent and so only the results of the last
    # filter will persist
    save_plots = None  #  "../output/CruncherResults/plots/"
    # Do we want to run this for all possible filters? If so, choose none,
    # otherwise specify the filter here as a list of tuples
    to_compare_filter = None
    """[
        ("AIM/CGE 2.0", "SSP1-19"),
        ("AIM/CGE 2.1", "TERL_15D_NoTransportPolicy"),
    ]"""
    years = range(2020, 2101, 10)
    # __________________________________end options_____________________________________

    assert len(crunchers_list) == len(crunchers_name_list)
    assert len(options_list) == len(crunchers_name_list)
    vars_to_crunch = [
        "Emissions|CO2|Energy and Industrial Processes", "Emissions|CO2|AFOLU"
    ]
    db_all = pyam.IamDataFrame(input_data).filter(region="World", year=years)
    db_all.filter(variable=vars_to_crunch + leaders, inplace=True)
    db_consistent = return_cases_which_consistently_split(
        db_all, leaders[0], vars_to_crunch
    )
    db_all = db_all.data
    db_all = pyam.IamDataFrame(db_all.iloc[[i in db_consistent for i in list(
        zip(db_all.model, db_all.scenario, db_all.region))]])

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
            vars_to_crunch,
            crunchers_name_list,
            crunchers_list,
            save_plots,
            leaders,
            options_list,
        )
        for filter_instance in all_possible_filters
    ]

    # Perform the loop
    with Pool(cpu_count() - 1) as pool:
        results_db = list(pool.map(_recalc_and_compare_results, all_args))
    #results_db = list(map(_recalc_and_compare_results, all_args))

    results_count = sum([result.notnull() for result in list(results_db)])
    overall_results = sum([result.fillna(0) for result in list(results_db)])
    overall_results = overall_results / results_count
    overall_results.to_csv(save_file)
    pd.concat(results_db).to_csv(save_workings)


def _recalc_and_compare_results(args):
    (
        one_filter,
        db_all,
        vars_to_crunch,
        crunchers_name_list,
        crunchers_list,
        save_plots,
        leaders,
        options_list,
    ) = args
    combo_filter = {"model": one_filter[0], "scenario": one_filter[1]}
    original_vals = db_all.filter(**combo_filter)
    input_to_fill = original_vals.filter(variable=leaders)
    results_db = pd.DataFrame(index=vars_to_crunch, columns=crunchers_name_list)
    # Remove all items that overlap directly with this
    db = db_all.filter(**combo_filter, keep=False)
    # Set up normalisation
    for cruncher_ind in range(len(crunchers_list)):
        # Initialise the object that holds the results
        cruncher_instance = crunchers_list[cruncher_ind](db)
        interpolated = cruncher_instance.infill_components(
            to_infill_df=input_to_fill, **options_list[cruncher_ind]
        )
        for var_inst in vars_to_crunch:
            originals = original_vals.filter(variable=var_inst).data.set_index("year")[
                "value"
            ]
            if originals.empty:
                print("No data available for {}".format(var_inst))
                continue

            interp_values = interpolated.filter(variable=var_inst).data.set_index("year")["value"]
            if originals.size != interp_values.size:
                print(
                    "Wrong number of values from cruncher {}: {}, not {}".format(
                        crunchers_name_list[cruncher_ind],
                        interp_values.size,
                        originals.size,
                    )
                )
                continue
            assert (
                interpolated["year"].size == interpolated["year"].unique().size * 2
            ), "The wrong number of years have returned values"
            # Calculate the RMS difference, Normalised by the spread of values
            norm_factor = pd.Series(index=interp_values.index, dtype=float)
            for year in norm_factor.index:
                norm_factor[year] = np.std(
                    db_all.filter(year=year, variable=var_inst).data["value"]
                )
            results_db[crunchers_name_list[cruncher_ind]][var_inst] = (
                np.nanmean(
                    (
                        (interp_values - originals)[norm_factor > 0]
                        / norm_factor[norm_factor > 0]
                    )
                    ** 2
                )
            ) ** 0.5
        print("Completed cruncher {}".format(crunchers_name_list[cruncher_ind]))
    return results_db


if __name__ == "__main__":
    main()
