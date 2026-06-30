import numpy as np
from numpy.polynomial.legendre import legval
from numpy.polynomial.polynomial import polyval
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from pint.fitter import WLSFitter
from pint.models import get_model
from pint.toa import get_TOAs
from pint.residuals import Residuals
from pint import logging
logging.setup("WARNING")
import astropy.units as u
from utils import plot_residuals, get_dmx_params, freeze_parameters, get_FD_delay
import sys

#-----------------------------------------------------------------------------------------------------------
# Set parameters up and load the data
#-----------------------------------------------------------------------------------------------------------

# Global parameters
PSR_name: str = "B1937+21"     # Name of the pulsar
method: str = "LIFD"           # Method you want to use. Options: FD, IFD, LIFD
plot_fits: bool = True         # Do you wanna plot the pre-fit and post-fit residuals?
simulations: bool = False      # Are you running simulations as opposed to real data?
order: int = 6                 # Order of the polynomial you want to use

# Input files
if simulations:
    parfile: str = f"./simulations/simplified_timing_model_pint.par"
    timfile: str = f"./simulations/timfile_freq_nofreqev.tim"
else:
    parfile: str = glob(f"./NANOGrav15yr_PulsarTiming_v2.1.0/narrowband/par/{PSR_name}_PINT_*.nb.par")[0]
    timfile: str = glob(f"./NANOGrav15yr_PulsarTiming_v2.1.0/narrowband/tim/{PSR_name}_PINT_*.nb.tim")[0]

#-----------------------------------------------------------------------------------------------------------
# This is where the fun begins
#-----------------------------------------------------------------------------------------------------------

for i, method in enumerate(["FD", "IFD", "LIFD"]):

    print(f"Running {method}")

    if simulations:
        timing_model = get_model(parfile, allow_tcb=True)  # Use allow_tcb because the simulations are made in Tempo2
        toas = get_TOAs(timfile, planets=True, model=timing_model, include_bipm=True)
        mask_pint = toas.get_mjds() < 58484.0 * u.day      # Mask later TOAs because the clock corrections are no good
        toas = toas[mask_pint]
    else:
        timing_model = get_model(parfile)                                        # Ecliptical coordiantes
        toas = get_TOAs(timfile, planets=True, ephem=timing_model.EPHEM.value)   # Load toas

    if plot_fits:
        '''
         # Set all of the DMX parameters to zero
        tmp_model = timing_model
        for DMX in tmp_model.components["DispersionDMX"].params:
            if DMX.startswith("DMX_"):
                getattr(tmp_model, DMX).value = 0.0
                getattr(tmp_model, DMX).uncertainty = 0.0 * getattr(tmp_model, DMX).units
                getattr(tmp_model, DMX).frozen = False

        # Freeze DMX windows without TOAs
        tmp_model.find_empty_masks(toas, freeze=True)

        # Now set the FD parameters to zero:
        for FD in tmp_model.components["FD"].params:
            getattr(tmp_model, FD).value = 0.0
            getattr(tmp_model, FD).uncertainty = 0.0 * getattr(tmp_model, FD).units
            getattr(tmp_model, FD).frozen = False
        '''
        # Calculate the pre-fit residuals
        res_pint = Residuals(toas, timing_model, subtract_mean=False).time_resids.to(u.us).value
        errs_pint = toas.get_errors().value
        freqs_pint = toas.get_freqs().value
        toas_mjd_pint = toas.get_mjds().value
        '''
        # Weighted mean from the good TOAs only
        weights_pint = 1.0 / errs_pint ** 2
        res_pint -= np.average(res_pint, weights=weights_pint)
        '''
        # Plot the pre-fit residuals
        fig_aux, ax_aux = plot_residuals(toas_mjd_pint, res_pint, freqs_pint)
        ax_aux.set_title("Pre-fit", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"./results/{method}_simulation_nofreqeq_residuals_prefit.png")
        plt.show()
        plt.close(fig_aux)

    # We'll use the variables to store the delay due to DM, profile evolution, and both
    new_dmx_dispersion_delays_us = np.zeros(toas.ntoas)
    prof_evol_delay_us = np.zeros(toas.ntoas)
    total_delay = np.zeros(toas.ntoas)

    #-----------------------------------------------------------------------------------------------------------
    # Change the DM so that the epoch-wise correcitons are smaller. See DM_search.py to see how I did this.
    #-----------------------------------------------------------------------------------------------------------
    change_dm = True
    if change_dm:
        DM_value = np.load(f"./results/{PSR_name}/new_DM.npy").item()
        DM_param = {"DM": (DM_value * timing_model.DM.units, 1, 0 * timing_model.DM.units)}

        # Change the DM value to the updated one
        for name, info in DM_param.items():
            par = getattr(timing_model, name)  # Get parameter object from name
            par.value = info[0]  # set parameter value
            if info[1] == 1:
                par.frozen = False  # Frozen means do not fit
            par.uncertainty = info[2]  # set parameter uncertainty

    # Fix the DM value so that it is not included in the timing fit
    getattr(timing_model, 'DM').frozen = False

    #-----------------------------------------------------------------------------------------------------------
    # Set up the IFD or LIFD model
    #-----------------------------------------------------------------------------------------------------------
    if method == "IFD" or method == "LIFD":
        timing_model.remove_component("FD")  # Remove the FD model

        # The LIFD component expects to define its λ→x mapping at setup(), which requires TOAs.
        # Therefore, we will attach TOAs to the model before adding LIFD:
        timing_model.toas = toas

        # Add the IFD or LIFD parameters
        if method == "IFD":
            from IFD_class import IFD
            timing_model.add_component(IFD(order=order))  # Attach the IFD component
            freeze_parameters(timing_model, ['IFD0'])   # AIC is telling us this term does not matter

        elif method == "LIFD":
            from LIFD_class import LIFD
            timing_model.add_component(LIFD(order=order))   # Attach the LIFD component

    #-----------------------------------------------------------------------------------------------------------
    # Fit the timing model
    #-----------------------------------------------------------------------------------------------------------
    f = WLSFitter(toas, timing_model)
    f.fit_toas()
    fitted_model = f.model

    # Plot the post-fit residuals
    if plot_fits:
        fig_aux, ax_aux = plot_residuals(toas_mjd_pint, f.resids.time_resids, freqs_pint)
        ax_aux.set_title(f"Post-Fit | {method}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"./results/{method}_simulation_nofreqeq_residuals_postfit.png")
        plt.show()
        plt.close(fig_aux)

    #-----------------------------------------------------------------------------------------------------------
    # Calculate the timing delays and save the coefficients
    #-----------------------------------------------------------------------------------------------------------
    freq_GHz = toas.get_freqs().to(u.GHz)    # Frequencies in GHz
    freq_rounded = np.round(freq_GHz, decimals=3).value
    unique_freqs = np.unique(freq_rounded)  # Find and return the sorted unique elements
    freq_GHz = unique_freqs * u.GHz

    # Set up the plotting
    sns.set_context("paper", font_scale=1.50, rc={"lines.linewidth": 2.5})
    fig, ax = plt.subplots(1, 1)
    ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
    colors = plt.colormaps['Paired'](range(8))
    colors = [colors[6], colors[7], colors[0], colors[1], colors[2], colors[3]]

    if method == "FD":
        # For plotting later
        x_var = np.sort(freq_GHz.value)
        ax.set_xlabel("$\\nu$ [GHz]")

        # Extract and save the FD coefficients
        FD_coeffs = [getattr(fitted_model, FD_param).value for FD_param in fitted_model.components['FD'].params]  # seconds
        np.save(f"./results/{PSR_name}/FD_coeffs.npy", FD_coeffs)

        # Calculate the delay due to pulse profile evolution as a function of frequency
        # Because of how I get_FD_delay is set up, this will automatically have units of microseconds
        prof_evol_delay_us = get_FD_delay(FD_coeffs, x_var)

    elif method == "IFD":
        # For plotting later
        x_var = np.sort(IFD.get_lambda_ns_from_freq(freq_GHz))  # Inverse frequencies
        ax.set_xlabel("$\lambda$")

        # Extract and save the IFD coefficients
        IFD_coeffs = [getattr(fitted_model, f"IFD{deg}").value for deg in range(0, order)]
        np.save(f"./results/{PSR_name}/IFD_coeffs.npy", IFD_coeffs)

        # Calculate the delay due to pulse profile evolution as a function of frequency
        # This will automatically have units of seconds because the coefficients have units of seconds
        prof_evol_delay_us = (polyval(x=x_var, c=IFD_coeffs) * u.second).to(u.us)

    elif method == "LIFD":
        # For plotting later
        lambdas_sec = LIFD.get_lambda_sec_from_freq(freq_GHz)  # Inverse frequencies
        lifd_comp = fitted_model.components["LIFD"]
        x_var = np.sort(lifd_comp.map_lambda_to_unit(lambdas_sec, lifd_comp.lmin, lifd_comp.lmax))
        ax.set_xlabel("x")

        # Extract and save the LIFD coefficients
        LIFD_coeffs = [getattr(fitted_model, f"LIFD{i}").value for i in range(0, order)]
        np.save(f"./results/{PSR_name}/LIFD_coeffs.npy", LIFD_coeffs)

        # Calculate the delay due to pulse profile evolution as a function of frequency
        prof_evol_delay_us = (legval(x_var, LIFD_coeffs) * u.second).to(u.us)

    else:
        print("Method not recognized")
        sys.exit(1)


    if simulations:
        # Calculate the dispersive delay as a function of frequency
        new_DMX_component = fitted_model.components["DispersionDMX"]
        new_dmx_dispersion_delays_us = new_DMX_component.DMX_dispersion_delay(toas).to(u.us)
        new_dmx_dispersion_delay_means = np.array([new_dmx_dispersion_delays_us.value[freq_rounded == f].mean() for f in unique_freqs])
        new_dmx_dispersion_delay_means -= np.mean(new_dmx_dispersion_delay_means)

        # Calculate the total chromatic delay
        total_delay = new_dmx_dispersion_delay_means + prof_evol_delay_us

        fig_aux, ax_aux = plt.subplots(1, 1)
        ax_aux.plot(unique_freqs, prof_evol_delay_us, label="Profile Evolution", ls='--', color="C0")
        ax_aux.plot(unique_freqs, new_dmx_dispersion_delay_means, label="DMX", ls=':', color="C1")
        ax_aux.plot(unique_freqs, total_delay, label="Total", ls='-', color="C2")
        ax_aux.set_xlabel("$\\nu$ [GHz]")
        ax_aux.set_ylabel("Delay [us]")
        plt.title(method)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"./results/{method}_simulation_nofreqeq_timing_delays.png")
        plt.show()
        plt.close(fig_aux)


    #-----------------------------------------------------------------------------------------------------------
    # Calculate the dispersive delay and save the DMX parameters
    #-----------------------------------------------------------------------------------------------------------
    DMX_params = get_dmx_params(fitted_model)
    DMX_params.to_pickle(f"./results/{PSR_name}/{method}_DMX.pkl")

    new_DMX_component = fitted_model.components["DispersionDMX"]
    new_dmx_dispersion_delays_us = new_DMX_component.DMX_dispersion_delay(toas).to(u.us)
    new_dmx_dispersion_delay_means = np.array([new_dmx_dispersion_delays_us.value[freq_rounded == f].mean() for f in unique_freqs])
    new_dmx_dispersion_delay_means -= np.mean(new_dmx_dispersion_delay_means)

    #-----------------------------------------------------------------------------------------------------------
    # Calculate and plot the total delay
    #-----------------------------------------------------------------------------------------------------------
    total_delay = prof_evol_delay_us.value + new_dmx_dispersion_delay_means
#    total_delay = total_delay.value
    total_delay -= np.mean(total_delay)

    # Sort the frequencies and timing delays
    #sort_idx = np.argsort(freq_GHz)
    #sorted_freqs = freq_GHz[sort_idx]
    #sorted_delays = total_delay[sort_idx]

    # Round frequencies to nearest channel to group identical frequencies
    freq_rounded = np.round(freq_GHz, decimals=3).value
    unique_freqs = np.unique(freq_rounded)  # Find and return the sorted unique elements

    # Detect gaps: define a gap as any jump larger than some threshold
    gap_threshold = 0.05  # GHz — adjust based on your channel spacing
    diffs = np.diff(unique_freqs)
    gap_indices = np.where(diffs > gap_threshold)[0] + 1  # indices where gaps start

    # Insert NaNs at gap locations
    freqs_with_nans = np.insert(unique_freqs, gap_indices, np.nan)

    total_delays_means = np.array([total_delay[freq_rounded == f].mean() for f in unique_freqs])
    total_delays_means_with_nans = np.insert(total_delays_means, gap_indices, np.nan)   # To avoid gaps in frequency coverage

    total_delays_stds = np.array([total_delay[freq_rounded == f].std() for f in unique_freqs])
    total_delays_stds_with_nans = np.insert(total_delays_stds, gap_indices, np.nan)

    prof_evol_delay_means = np.array([prof_evol_delay_us.value[freq_rounded == f].mean() for f in unique_freqs])
    prof_evol_delay_means -= np.mean(prof_evol_delay_means)
    prof_evol_delay_means_with_nans = np.insert(prof_evol_delay_means, gap_indices, np.nan)

#    new_dmx_dispersion_delay_means = np.array([new_dmx_dispersion_delays_us.value[freq_rounded == f].mean() for f in unique_freqs])
#    new_dmx_dispersion_delay_means -= np.mean(new_dmx_dispersion_delay_means)
    new_dmx_dispersion_delay_means_with_nans = np.insert(new_dmx_dispersion_delay_means, gap_indices, np.nan)

    # Plot the total chromatic delay
    ax.plot(freqs_with_nans, total_delays_means_with_nans, ls='-', color=colors[2*i+1], label=f"{method}", alpha=0.6)
    ax.fill_between(freqs_with_nans,
                    total_delays_means_with_nans - total_delays_stds_with_nans,
                    total_delays_means_with_nans + total_delays_stds_with_nans, color=colors[2*i+1], alpha=0.1)

    # Plot the delays due to the individual components
    ax2.plot(freqs_with_nans, prof_evol_delay_means_with_nans, ls='--', color=colors[2*i])
    ax2.plot(freqs_with_nans, new_dmx_dispersion_delay_means_with_nans , ls=':', color=colors[2*i])

ax.set_xlabel("$\\nu$ [GHz]")
ax.set_ylabel("Total delay [$\mu$s]")
ax2.set_ylabel("Individual components delays [$\mu$s]")

ax.plot([], [], ls='-', color='black', label=f"Total delay")
ax.plot([], [], ls='--', color='black', alpha=0.3, label=f"Profile evolution")
ax.plot([], [], ls=':', color='black', alpha=0.3, label=f"DMX")

ax.legend(ncol=2)
plt.tight_layout()
plt.savefig(f"./figures/{PSR_name}_profile_evl_and_DMX.pdf")
plt.show()
