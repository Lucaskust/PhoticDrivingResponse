"""
This is the main module that calls all made functions.
"""
from patients import parse_args, patient_files, eeg, save_pickle_results, filter_files, add_patients, sort_results, sync
from power import Power
from phase import Phase
# from analytics import stats_base_power, stats_power, stats_plv
# from classification import Classifier

if __name__ == "__main__":
    # Reading data
    trial_map, time_map, args = parse_args()
    pt_files = patient_files(trial_map, args)
    skipped, complete_power, complete_plv = [], [], []
    all_mean_plv = {}
    N = 0
    PASSBAND = [0.5, 100]
    FOLDER_POWER = "results_POWER"
    FOLDER_PLV = "results_PLV"

    for pt_file in pt_files:
        N += 1
        print(f"--- Processing file {N}/{len(pt_files)}.")

        # Power calculation
        try:
            raw = eeg(pt_file, PASSBAND, occi=True, plot=False)
            power = Power(PASSBAND, raw).run()
            save_pickle_results(power, pt_file, FOLDER_POWER, feat="power")

        except Exception as e: # pylint: disable=broad-except
            print(f"Skipping power calculation of {pt_file.name} due to error: {e}")
            skipped.append((pt_file.name, "Power", str(e)))
            continue

        # PLV calculation
        try:
            plv_stim, plv_base = Phase(PASSBAND, raw).run()
            save_pickle_results({"stim": plv_stim, "base": plv_base}, pt_file, FOLDER_PLV, feat="plv")

        except Exception as e: # pylint: disable=broad-except
            print(f"Skipping phase calculation of {pt_file.name} due to Phase error: {e}")
            skipped.append((pt_file.name, "PLV", str(e)))
            continue
        # Na het verwerken van alle .cnt-bestanden:
    # complete_power = filter_files(FOLDER_POWER, time_map, args, feat="power")
    # complete_plv   = filter_files(FOLDER_PLV, time_map, args, feat="plv")
    sync(FOLDER_POWER, FOLDER_PLV, "results_incomplete")

    if skipped:
        print(f"All files processed. Skipped files: {skipped}")
    else:
        print("All files processed. No skipped files.")

    print(f"Complete files for {set(complete_power) & set(complete_plv)}")

    print("Feature-extractie klaar. Power/PLV zijn opgeslagen in 'results_POWER' en 'results_PLV'.")
    # Hier stoppen we voorlopig, zodat er geen statistiek/ML wordt gedraaid op onvolledige data.
    import sys
    sys.exit(0)

    # Alles hieronder blijft uit / mag weg:
    # responder_ids = {...}
    # stats_base_power(...)
    # df_power = stats_power(...)
    # df_plv = stats_plv(...)
    # ...


    # # Statistics
    # responder_ids = {"2", "10", "11", "17", "21", "22", "32", "40", "46", "48", "51", "57", "63"}
    # stats_base_power(FOLDER_POWER, paired=True, save=True) # Power baseline
    # df_power = stats_power(responder_ids, FOLDER_POWER, paired=True, save=True, plot=False)
    # df_plv = stats_plv(responder_ids, FOLDER_PLV, paired=True, save=True, plot=False)

    # # Extra files
    # N = 0
    # recovered_files = add_patients(args, set(complete_power) & set(complete_plv))
    # for pt_file in recovered_files:
    #     N += 1
    #     print(f"--- Processing file {N}/{len(recovered_files)}.")
    #     try:
    #         raw = eeg(pt_file, PASSBAND, occi=True, plot=False)
    #         power = Power(PASSBAND, raw).run()
    #         save_pickle_results(power, pt_file, "results_incomplete", feat="power")

    #         plv_stim, plv_base = Phase(PASSBAND, raw).run()
    #         save_pickle_results({"stim": plv_stim, "base": plv_base}, pt_file, "results_incomplete", feat="plv")

    #     except Exception as e: # pylint: disable=broad-except
    #         print(f"Skipping recovery of {pt_file.name} due to error: {e}")

    # sort_results("results_incomplete", "results_incomplete/power", "results_incomplete/plv")
    # df_power_incomplete = stats_power(responder_ids, "results_incomplete/power", paired=True, save=False, plot=False)
    # df_plv_incomplete = stats_plv(responder_ids, "results_incomplete/plv", paired=True, save=False, plot=False)
    # sync("results_incomplete/power", "results_incomplete/plv", "results_incomplete")

    # # Classification
    # pipeline = Classifier(df_power=df_power, df_plv=df_plv)
    # pipeline.run(task="abc")
    # pipeline.classify_new_data(df_power_incomplete, df_plv_incomplete, task="abc")
