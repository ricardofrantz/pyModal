import argparse

#!/usr/bin/env python3
"""Unified command-line interface for all modal analyses."""

from bmsd import BSMDAnalyzer
from configs import DEFAULT_DATA_FILE
from dmd import DMDAnalyzer
from pod import PODAnalyzer
from spod import SPODAnalyzer
from utils import auto_detect_weight_type, print_summary


def run_pod(data_file, prep, compute, plot):
    analyzer = PODAnalyzer(file_path=data_file, spatial_weight_type=auto_detect_weight_type(data_file))
    run_all = not (prep or compute or plot)
    if run_all or prep:
        analyzer.load_and_preprocess()
    if run_all or compute:
        if analyzer.data == {}:
            analyzer.load_and_preprocess()
        analyzer.perform_pod()
        analyzer.save_results()
    if run_all or plot:
        if analyzer.eigenvalues.size == 0:
            analyzer.load_results()
        analyzer.plot_eigenvalues()
        analyzer.plot_modes_pair_detailed()
        analyzer.plot_modes_grid(energy_threshold=99.5)
        analyzer.plot_time_coefficients()
        analyzer.plot_cumulative_energy()
        analyzer.plot_reconstruction_error()
        analyzer.plot_reconstruction_comparison()
    if run_all:
        pass
    print_summary("POD", analyzer.results_dir, analyzer.figures_dir)
    analyzer.release_memory()


def run_spod(data_file, prep, compute, plot):
    analyzer = SPODAnalyzer(
        file_path=data_file,
        nfft=256,
        overlap=0.5,
        spatial_weight_type=auto_detect_weight_type(data_file),
    )
    run_all = not (prep or compute or plot)
    if run_all or prep:
        analyzer.load_and_preprocess()
        analyzer.compute_fft_blocks()
    if run_all or compute:
        if analyzer.qhat.size == 0:
            analyzer.load_and_preprocess()
            analyzer.compute_fft_blocks()
        analyzer.perform_spod()
        analyzer.save_results()
    if run_all or plot:
        if analyzer.eigenvalues.size == 0:
            analyzer.load_results()
        analyzer.plot_eigenvalues_v2()
        analyzer.plot_modes()
        analyzer.plot_cumulative_energy()
        analyzer.plot_time_coeffs()
        analyzer.plot_reconstruction_error()
        analyzer.plot_eig_complex_plane()
    if run_all:
        pass
    print_summary("SPOD", analyzer.results_dir, analyzer.figures_dir)
    analyzer.release_memory()


def run_dmd(data_file, prep, compute, plot):
    analyzer = DMDAnalyzer(
        file_path=data_file,
        spatial_weight_type=auto_detect_weight_type(data_file),
    )
    run_all = not (prep or compute or plot)
    if run_all or prep:
        analyzer.load_and_preprocess()
    if run_all or compute:
        if analyzer.data == {}:
            analyzer.load_and_preprocess()
        analyzer.perform_dmd()
        analyzer.save_results()
    if run_all or plot:
        if analyzer.eigenvalues.size == 0:
            analyzer.load_results()
        analyzer.plot_eigenspectra()
        analyzer.plot_modes_detailed()
        analyzer.plot_time_coefficients()
        analyzer.plot_cumulative_energy()
        analyzer.plot_reconstruction_error()
    if run_all:
        pass
    print_summary("DMD", analyzer.results_dir, analyzer.figures_dir)
    analyzer.release_memory()


def run_bsmd(data_file, prep, compute, plot):
    analyzer = BSMDAnalyzer(
        file_path=data_file,
        nfft=128,
        overlap=0.5,
        spatial_weight_type=auto_detect_weight_type(data_file),
    )
    run_all = not (prep or compute or plot)
    if run_all or prep:
        analyzer.load_and_preprocess()
        analyzer.compute_fft_blocks()
    if run_all or compute:
        if analyzer.qhat.size == 0:
            analyzer.load_and_preprocess()
            analyzer.compute_fft_blocks()
        analyzer.perform_bsmd()
        analyzer.save_results()
    if run_all or plot:
        if analyzer.eigenvalues.size == 0:
            analyzer.load_results()
        analyzer.plot_modes()
    if run_all:
        pass
    print_summary("BSMD", analyzer.results_dir, analyzer.figures_dir)
    analyzer.release_memory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified pyModal runner")
    parser.add_argument("--pod", action="store_true", help="Run POD")
    parser.add_argument("--spod", action="store_true", help="Run SPOD")
    parser.add_argument("--dmd", action="store_true", help="Run DMD")
    parser.add_argument("--bsmd", action="store_true", help="Run BSMD")
    parser.add_argument("--prep", action="store_true", help="Only preprocess data")
    parser.add_argument("--compute", action="store_true", help="Only compute analysis")
    parser.add_argument("--plot", action="store_true", help="Only plot results")
    parser.add_argument("--data", default=DEFAULT_DATA_FILE, help="Path to data file")
    args = parser.parse_args()

    analyses = []
    if args.pod:
        analyses.append("pod")
    if args.spod:
        analyses.append("spod")
    if args.dmd:
        analyses.append("dmd")
    if args.bsmd:
        analyses.append("bsmd")
    if not analyses:
        analyses = ["pod", "dmd", "spod", "bsmd"]

    for analysis in analyses:
        if analysis == "pod":
            run_pod(args.data, args.prep, args.compute, args.plot)
        elif analysis == "spod":
            run_spod(args.data, args.prep, args.compute, args.plot)
        elif analysis == "dmd":
            run_dmd(args.data, args.prep, args.compute, args.plot)
        elif analysis == "bsmd":
            run_bsmd(args.data, args.prep, args.compute, args.plot)
