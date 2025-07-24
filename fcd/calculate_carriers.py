from dataclasses import dataclass
from numpy import array
import numpy as np
from pyfftw.interfaces.scipy_fftpack import fft2, ifft2

from find_peaks import find_peaks
from kspace import kvec

# complex conjugate of the inverse fft of the masked i_ref_fft
def ccsgn(i_ref_fft, mask):
    return np.conj(ifft2(i_ref_fft * mask))

@dataclass
class Carrier:
    k_loc: array
    krad: float
    mask: array
    ccsgn: array

def calculate_carriers(i_ref):
    kr, ku = find_peaks(i_ref) # k-space coords
    
    peak_radius = np.sqrt(np.sum((kr - ku)**2)) / 2
    i_ref_fft = fft2(i_ref)

    def create_mask(shape, kc, krad):
        r, c = shape
        kx, ky = np.meshgrid(kvec(c), kvec(r))
        # Build the circular mask in k-space centered on kc = [kx, ky]
        return ((kx - kc[0])**2 + (ky - kc[1])**2) < krad**2

    carriers = []
    for k_loc in [kr, ku]:
        mask = create_mask(i_ref.shape, k_loc, peak_radius)
        carrier = Carrier(k_loc=np.array(k_loc),
                          krad=peak_radius,
                          mask=mask,
                          ccsgn=ccsgn(i_ref_fft, mask))
        carriers.append(carrier) 
    
    return carriers
