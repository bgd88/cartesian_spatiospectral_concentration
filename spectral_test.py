import numpy as np
import matplotlib.pyplot as plt
import sleppy
from scipy import signal

# The Fourier Analysis in this script follows the analysis of Perron et al 2008, and Numerical Recipes

def hann2d(nx, ny):
    """
    Calcualtes coeficients for an elliptical Hann (raised cosine) window 
    [H] and the summed square of the weighting coefficients [Wss].
    Wss is used in the normalization of the power spectrum.
    See Numerical Recipes 13.4   
    """
    a, b = (nx+1)/2, (ny+1)/2 # Matrix Centroid
    X, Y = np.meshgrid( np.arange(nx)+1., np.arange(ny)+1. ) # Index Matrix
    theta = (X==a)*np.pi/2. + (X!=a)*np.arctan2((Y-b),(X-a)) # Angular polar coordinate
    r = np.sqrt((X-a)**2 + (Y-b)**2) # radial polar coordinate
    a2, b2 = a**2, b**2
    rprime = np.sqrt(  a2*b2*np.power((b2*(np.cos(theta)**2) + a2*(np.sin(theta)**2)), -1.) )  # ellipse radius
    hanncoef = (r < rprime)*(0.5*(1.+np.cos(np.pi*r/rprime))) # Window coefs
    Wss = np.sum(hanncoef**2) # summed square of
    return hanncoef, Wss 


def calculate_radial_frequencies(nx, ny, dx, dy):
    ic, jc = nx/2, ny/2  # indices of zero freqency
    cols, rows = np.meshgrid( np.arange(nx), np.arange(ny) ) 
    
    # calculate the frequency increments: frequency goes from zero to 1/(2*dx)
    # (Nyquist in x direction) in Nx/2 increments
    dfx, dfy = 1./(dx*nx), 1./(dy*ny)

    # Frequency Matrix
    radial_frequencies = np.sqrt( (dfx*(cols-xc))**2 + (dfy*(rows-jc))**2 )


def fft2d(data, dx, dy):
    """
    This function performs the discrete 2D FFT, and retruns the 2D Fourier Spectra
    and the normalized power spectrum.

    The input data is detrended, and a Hanning window is applied as pre-processing.
    """
    ny, nx = data.shape

    # Detrend data
    pre_processed_data = data #signal.detrend(data)

    # Create Hanning Taper
    hann, Wss = hann2d(nx, ny)

    # Perform 2D FFT
    F = np.fft.fft2(pre_processed_data)
    F = np.fft.fftshift(F)  # shift to put the zero wavenumer at the center
    F[ny/2, nx/2] = 0

    # Calc mag. phase and DFT periodigram
    F_mag = np.abs(F)
    F_phase = np.angle(F)
    F_power = np.abs(F)**2/(nx*ny*Wss) # Normalization corrects for the amplitude reduction by the windowing
                                              # function. See Numerical Recipes 13.4

    # Calculate Radial Frequencies 
    #radial_freq = calculate_radial_frequencies(nx, ny, dx, dy)
    
    # Creaite sorted non-redundent vector of freq and power
    #power_vec = F[:,range(nx/2)]
    #freq_vec = radial_freq[:, range(nx/2)]
    #freq_vec[range(ny/2+1, ny), nx/2]
    return F_power


def slepian_multitaper(xgrid, ygrid, data, basis):
    nx, ny = data.shape
    
    power_spectra = np.zeros_like(data)
    for weight, taper in basis:
        tapered_data = taper(xgrid, ygrid)*data

        F = np.fft.fft2(tapered_data)
        current_power = np.abs(F)**2
        power_spectra += weight*current_power
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        data_plot = ax1.imshow(tapered_data)
        ax1.set_title("Synthetic_data")
        ps = ax2.imshow(np.log(current_power))
        plt.colorbar(ps)
        ax2.set_title(" Power Spectrum - w/Slepian Multi-Tapper")
        plt.show()
 
    return power_spectra/len(basis) 

                                        
def calculate_radial_frequencies(nx, ny, dx, dy):
    ic, jc = nx/2, ny/2  # indices of zero freqency
    cols, rows = np.meshgrid( np.arange(nx), np.arange(ny) ) 
    
    # calculate the frequency increments: frequency goes from zero to 1/(2*dx)
    # (Nyquist in x direction) in Nx/2 increments
    dfx, dfy = 1./(dx*nx), 1./(dy*ny)

    # Frequency Matrix
    radial_frequencies = np.sqrt( (dfx*(cols-xc))**2 + (dfy*(rows-jc))**2 )

# Initialize domain and grids
Lx = Ly = 2.*np.pi
Nx, Ny = 256, 256
dx, dy = Lx/Nx, Ly/Ny
x, y = np.linspace(0., Lx, Nx), np.linspace(0., Ly, Ny)
xgrid, ygrid = np.meshgrid(x,y)

# Create circular subdomain
R = 2.
domain = sleppy.Disc( (Lx, Ly), (Lx/2., Ly/2.), R)

# Create synthetic data
k_x, k_y = 10., 10.
synthetic_data = np.cos(k_x*xgrid)*np.cos(k_y*ygrid)

power_spectrum = fft2d(synthetic_data, dx, dy)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
data_plot = ax1.imshow(synthetic_data)
ax1.set_title("Synthetic_data")
ps = ax2.imshow(np.log(power_spectrum))
plt.colorbar(ps)
ax2.set_title(" Power Spectrum - w/Hanning Tapper")
plt.show()

synthetic_data[~domain.in_subdomain(xgrid, ygrid)] = 0
power_spectrum = fft2d(synthetic_data, dx, dy)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
data_plot = ax1.imshow(synthetic_data)
ax1.set_title("Synthetic_data")
ps = ax2.imshow(np.log(power_spectrum))
plt.colorbar(ps)
ax2.set_title(" Power Spectrum - w/Hanning Tapper")
plt.show()


basis = sleppy.compute_slepian_basis( domain, 4, basis_function_type='interpolated')

multi_taper_spectrum = slepian_multitaper(xgrid, ygrid, synthetic_data, basis)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
data_plot = ax1.imshow(synthetic_data)
ax1.set_title("Synthetic_data")
ps = ax2.imshow(np.log(multi_taper_spectrum))
plt.colorbar(ps)
ax2.set_title(" Power Spectrum - w/Slepian Multi-Tapper")
plt.show()
