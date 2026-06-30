import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.units import Quantity
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.linalg import cho_factor
from numpy.polynomial.polynomial import Polynomial
from pint.residuals import Residuals
from pypulse.utils import weighted_moments
from pypulse.par import Par
import astropy.units as u
from dataclasses import dataclass



@dataclass
class DataObject:
    PSR_name: str
    dmx_ranges: np.matrix
    max_inv_freq: float
    min_inv_freq: float
    logdet_C: float
    Ndiag: np.array
    U: np.matrix
    Sigma_cf: np.matrix
    xvals: list[np.ndarray]
    freqs: list[np.ndarray]
    resids: list[np.ndarray]


def get_dmx_ranges(model, mjds):
    """
    Return an array of the MJD range for each DMX parameter corresponding to a given set of observations_in_window
    """

    dmx_model = model.components['DispersionDMX']  # names of the DMX_xxxx parameters

    # List containing the MJDs of the beginning and end of each DMX window
    DMXR = np.fromiter(((dmx_model._parent[f"DMXR1_{idx:04d}"].value, dmx_model._parent[f"DMXR2_{idx:04d}"].value)
                        for idx in dmx_model.get_indices()), dtype=np.dtype((float, 2)))

    # Mask all the DMX windows that do not contain any TOAs
    mask = np.any(((DMXR[:, 0, None] <= mjds) & (mjds <= DMXR[:, 1, None])), axis=1)
    
    return DMXR[mask]


def get_dmx_observations(observations, low_mjd, high_mjd):
    """
    Return an array for selecting TOAs from toas in a DMX range.

    toas is a PINT TOA object of TOAs in the DMX bin.
    low_mjd is the left edge of the DMX bin.
    high_mjd is the right edge of the DMX bin.
    strict_inclusion=True if TOAs exactly on a bin edge are not in the bin for
        the implemented DMX model.
    """

    mjds = observations.get_mjds().value
    mask = (low_mjd < mjds) & (mjds < high_mjd)

    return observations[mask]


def map_domain(frequencies, max_inv_freq, min_inv_freq):

    lambdas = np.power(frequencies, -1.0)                                # Inverse of the frequencies
    x_aux_values = (lambdas - min_inv_freq)/(max_inv_freq-min_inv_freq)  # Between 0 and 1
    x_values = np.subtract(np.multiply(x_aux_values, 2.0), 1.0)          # Between -1 and 1

    return np.asarray(x_values, dtype=np.float64)

def reverse_mapping(x_values, max_inv_freq, min_inv_freq):

    aux = np.divide(np.add(x_values, 1.0), 2.0)                 # Between 0 and 1
    lambdas = aux * (max_inv_freq-min_inv_freq) + min_inv_freq  # Inverse of the frequencies, in GHz^(-1)
    frequencies = np.power(lambdas, -1.0)                       # In GHz

    return frequencies


def get_data(psr_name, toas, timing_model):

    """Given TOAs, extract broadband observations and select DMX windows with both frequency bands."""

    # Filter out narrowband receivers
    backends = np.array([toas.table["flags"][obs]["be"] for obs in range(len(toas.table["flags"]))])
    broadband_TOAs = toas[~np.isin(backends, ["GASP", "ASP"])]

    # Extract relevant information from the broadband TOAs
    broadband_mjds = broadband_TOAs.get_mjds().value                     # MJDs of the broadband TOAs
    broadband_dmx_ranges = get_dmx_ranges(timing_model, broadband_mjds)  # Find the DMX windows with broadband TOAs
    freqs_GHz = broadband_TOAs.get_freqs().to(u.GHz).value               # Frequencies in GHz
    inverse_freqs = np.power(freqs_GHz, -1)                              # Inverse frequencies in GHz^(-1)
    max_inv_freq, min_inv_freq = np.amax(inverse_freqs), np.amin(inverse_freqs)
    xvals = map_domain(freqs_GHz, max_inv_freq, min_inv_freq)            # Inverse of the frequencies, mapped to [-1, 1]

    # Get rid of the DMX and FD parameters to create the simplified timing model
    timing_model.remove_component("DispersionDMX")
    timing_model.remove_component("FD")

    # Calculate the residuals
    res_object = Residuals(broadband_TOAs, timing_model)
    residuals = np.asarray(res_object.time_resids.to(u.us).value, dtype=np.float64)

    # Find the matrices that we will use to calculate the residuals and covariance matrix
    Ndiag = res_object.model.scaled_toa_uncertainty(broadband_TOAs).to_value(u.s) ** 2
    U = res_object.model.noise_model_designmatrix(res_object.toas)
    Phidiag = res_object.model.noise_model_basis_weight(res_object.toas)

    # See Eq. 13 of https://iopscience.iop.org/article/10.3847/1538-4357/ad59f7/pdf
    Sigma = np.diag(1.0 / Phidiag) + (U.T / Ndiag) @ U
    Sigma_cf = cho_factor(Sigma)

    logdet_N = np.sum(np.log(Ndiag))
    logdet_Phi = np.sum(np.log(Phidiag))
    _, logdet_Sigma = np.linalg.slogdet(Sigma.astype(float))

    logdet_C = np.asarray(logdet_N + logdet_Phi + logdet_Sigma, dtype=np.float64)

    # Create window masks using array operations
    window_masks = [(broadband_mjds > window[0]) & (broadband_mjds < window[1]) for window in broadband_dmx_ranges]

    # Apply masks to create grouped arrays using list comprehensions
    resids = [residuals[mask] for mask in window_masks]
    xvals = [xvals[mask] for mask in window_masks]
    freqs = [freqs_GHz[mask] for mask in window_masks]

    return DataObject(PSR_name=timing_model.PSR.value, dmx_ranges=broadband_dmx_ranges,
                    max_inv_freq=max_inv_freq, min_inv_freq=min_inv_freq,
                    logdet_C=logdet_C, Ndiag=Ndiag, U=U, Sigma_cf=Sigma_cf,
                    xvals=xvals, freqs=freqs, resids=resids)  # , resids_errs=valid_resids_errs)


def plot_all_coeffs_fit(PSR_name, df):
    windows_centers = df["DMXR1"] + (df["DMXR2"] - df["DMXR1"]) / 2.0

    sns.set_style("ticks")
    sns.set_context("paper", font_scale=3.0)
    fig, ax = plt.subplots(nrows=6, ncols=1, figsize=(12, 24), sharex=True,
                           gridspec_kw={'hspace': 0})
    fig.suptitle(PSR_name + " - Monomial Coefficients in $x$ \n $t_\\nu = b_0 + b_1 x + b_2 x^2 + b_3 x^3 + b_4 x^4 + b_5 x^5$")


    means = df[['a0', 'a1', 'a2', 'a3', 'a4', 'a5']].mean(axis=0)
    print(f"a1 = {means['a1']}")
    print(f"a3 = {means['a3']}")
    print(f"a5 = {means['a5']}")

    # Plot and label each subplot
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    for i in range(6):
        ax[i].scatter(windows_centers, df[f'a{i}'], color=colors[i])
        ax[i].axhline(y=means[f'a{i}'], color='black', lw=4, linestyle='--')
        ax[i].set_ylabel(f"$b_{i} [ns]$")
        ax[i].grid(True)  # Add grid
        ax[i].label_outer()  # Hide inner x labels and ticks

        ax[i].text(0.2, 0.1, f'Mean = {round(means[f"a{i}"], 4)}', horizontalalignment='center', verticalalignment='center',
                 transform=ax[i].transAxes)

    ax[5].set_xlabel("Window Center [MJD]")

    plt.tight_layout()
    plt.savefig('./' + PSR_name + '_results.png')
    plt.show()

    return


def epoch_scrunch(toas, data=None, errors=None, epochs=None, decimals=0, getdict=False, weighted=False, harmonic=False):
    if epochs is None:
        epochsize = 10 ** (-decimals)
        bins = np.arange(np.around(min(toas), decimals=decimals) - epochsize,
                         np.around(max(toas), decimals=decimals) + 2 * epochsize,
                         epochsize)  # 2 allows for the extra bin to get chopped by np.histogram
        freq, bins = np.histogram(toas, bins)
        validinds = np.where(freq != 0)[0]

        epochs = np.sort(bins[validinds])
        diffs = np.array(list(map(lambda x: np.around(x, decimals=decimals), np.diff(epochs))))
        epochs = np.append(epochs[np.where(diffs > epochsize)[0]], [epochs[-1]])
    else:
        epochs = np.array(epochs)
    reducedTOAs = np.array(list(map(lambda toa: epochs[np.argmin(np.abs(epochs - toa))], toas)))

    if data is None:
        return epochs

    Nepochs = len(epochs)

    if weighted and errors is not None:
        averaging_func = lambda x, y: weighted_moments(x, 1.0 / y ** 2, unbiased=True, harmonic=harmonic)
    else:
        averaging_func = lambda x, y: (np.mean(x), np.std(y))  # is this correct?

    if getdict:
        retval = dict()
        retvalerrs = dict()
    else:
        retval = np.zeros(Nepochs)
        retvalerrs = np.zeros(Nepochs)
    for i in range(Nepochs):
        epoch = epochs[i]
        inds = np.where(reducedTOAs == epoch)[0]
        if getdict:
            retval[epoch] = data[inds]
            if errors is not None:
                retvalerrs[epoch] = errors[inds]
        else:
            if errors is None:
                retval[i] = np.mean(data[inds])  # this is incomplete
                retvalerrs[i] = np.std(data[inds])  # temporary
            else:
                retval[i], retvalerrs[i] = averaging_func(data[inds], errors[inds])
    #            print data[inds],errors[inds]
    if getdict and errors is None:  # is this correct?
        return epochs, retval
    return epochs, retval, retvalerrs



def get_FD_curve_values(p, freqs, DM0=0.0):

    FDfunc = p.getFDfunc()   # Frequency is in GHz, returns values of microseconds
    if FDfunc is None:
        return

    DM = p.getDM()
    ts, dmx, errs, R1s, R2s, _, _ = p.getDMX(full_output=True)
    F1s = np.amin(freqs)
    F2s = np.amax(freqs)
#    F1 = np.min(F1s)/1000.0 #in GHz
#    F2 = np.max(F2s)/1000.0

    fs = np.arange(F1s, F2s, 0.001)

#    shift = -K*((DM+dmx[0])-DM0)/fs**2
    shift = 0.0

    ys = FDfunc(fs) + shift
    ys -= np.mean(ys)

    return fs, ys


def plot_FD_curve(PSR_name, parfile, data_obj, samples):
    # Plot the FD curves
    fig, ax = plt.subplots()
    fig.suptitle(PSR_name)
    a1a3a5 = np.median(samples, axis=0)  # Calculate the maximum posterior coefficients
    poly = Polynomial([0.0, a1a3a5[0], 0.0, a1a3a5[1], 0.0, a1a3a5[2]])  # Construct the power series polynomial
    x_vals = np.arange(-1.0, 1.0, 0.001)  # Create values of the normalized inverse frequency between -1 and 1
    ys = poly(x_vals)  # Evaluate the power series polynomial at those inverse frequencies
    ys -= np.mean(ys)

    # Transform the inverse frequencies to normal frequencies (in GHz)
    frequencies = reverse_mapping(x_vals, data_obj.max_inv_freq, data_obj.min_inv_freq)
    ax.plot(frequencies, ys * 10**(-3), label="$a_1 x + a_3 x^3 + a_5 x^5$")

    # FD model
    p = Par(parfile, numwrap=float)
    DM = p.getDM()

    # Frequencies and delays for the model as it is
    fs, ys_FD = get_FD_curve_values(p, frequencies, DM0=DM)
    ax.plot(fs, ys_FD * 10**(-3), 'k', label="NG15's FD model")
    F1, F2 = frequencies[0], frequencies[-1]
    Fdiff = F2 - F1
    ax.set_xlim(F1 - 0.1 * Fdiff, F2 + 0.1 * Fdiff)

    ax.set_xlabel(r"Frequency (GHz)")
    ax.set_ylabel(r"Residual ($\mu$s)")

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./results/{PSR_name}/{PSR_name}_FD_curve.png")
    plt.show()

def plot_residuals(toas, residuals, freqs, toaerrs=None, title=None, errs_smoothened=False):
    toas = np.asarray(toas)
    residuals = np.asarray(residuals)

    if freqs is not None:
        freqs = np.asarray(freqs)

    if toaerrs is not None:
        toaerrs = np.asarray(toaerrs)

    if toaerrs is not None and type(errs_smoothened) == int:
        errs_std = np.std(toaerrs)
        indexes = np.where(toaerrs < errs_smoothened * errs_std)

        residuals = residuals[indexes]
        toas = toas[indexes]
        toaerrs = toaerrs[indexes]

        if freqs is not None:
            freqs = freqs[indexes]

    # Convert MJD to fractional years and normalize to start at year 0
    years = 1858.878 + toas / 365.25
#    years -= years.min()

    # Create figure and axes explicitly so Matplotlib knows where to place the colorbar
    fig, ax = plt.subplots(figsize=(10, 6))

    if freqs is not None:
        print("Using freq!")

        vmin = freqs.min() - freqs.min() * 0.1
        vmax = freqs.max() + freqs.max() * 0.1

        cmap = LinearSegmentedColormap.from_list("", ["blue", "yellow"])
        norm = Normalize(vmin=vmin, vmax=vmax)

        colors = cmap(norm(freqs))

        if toaerrs is not None:
            ax.errorbar(years, residuals, yerr=toaerrs, fmt="none", ecolor=colors, elinewidth=1.1)
        else:
            ax.errorbar(years, residuals, yerr=toaerrs, fmt=".", ecolor="blue", label="Timing Residuals")

        sc = ax.scatter(years, residuals, c=freqs, cmap=cmap, norm=norm, label="Timing Residuals")
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Frequency (MHz)", fontsize=12)

    elif toaerrs is not None:
        print("Using errs!")
        ax.errorbar(years, residuals, yerr=toaerrs, fmt=".", ecolor="blue", label="Timing Residuals")

    else:
        print("Not using freqs or errs.")

        ax.scatter(years, residuals, edgecolor="k", label="Timing Residuals")

    # Reference line at zero
    ax.axhline(0, color="red", linestyle="--", linewidth=1)

    # Labels and title
    #ax.set_xlabel("Years since first observation", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Residuals (μs)", fontsize=12)

    if title is not None:
        ax.set_title(title, fontsize=14)

    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()

    return fig, ax

def freeze_parameters(model, params_to_freeze):
    """Zero out, freeze, and set uncertainty to 0 for a list of parameter names."""
    for p in params_to_freeze:
        param = getattr(model, p)
        param.value = 0.0
        param.uncertainty_value = 0.0
        param.frozen = True



def get_FD_delay(FD_params, freqs_GHz):
    """
    Returns a function that provides the timing delays as a function of observing frequency in units of microseconds
    """
    FD_params = FD_params[::-1]  # We need to invert the order because np.polyval uses p[0]*x**(N-1)+p[1]*x**(N-2)+...
    FD_params = np.concatenate((FD_params, [0]))  # https://numpy.org/doc/stable/reference/generated/numpy.polyval.html
    FD_func = lambda nu: 1e6 * np.polyval(FD_params, np.log(nu))  # Function that gives the timing delay
    FD_delays = FD_func(freqs_GHz)  * u.us  # nu in GHz, returns values in microseconds
    return FD_delays


def get_dmx_params(timing_model):

    dmx_params = timing_model.components["DispersionDMX"].params
    names = []
    values = np.empty(len(dmx_params))
    errors = np.full(len(dmx_params), np.nan) * u.pc / u.cm**3

    for i, par in enumerate(dmx_params):
        names.append(getattr(timing_model, par).name)
        values[i] = getattr(timing_model, par).value

        if type(getattr(timing_model, par).uncertainty) is Quantity:
            errors[i] = getattr(timing_model, par).uncertainty
        else:
            errors[i] = np.nan * u.pc / u.cm**3

    return pd.DataFrame({'Value': values, 'Error': errors}, index=names)