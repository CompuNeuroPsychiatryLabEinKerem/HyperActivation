# CiftiUtils

import nibabel as nb

def setNumTpForCiftiHdr(chdr, numTrs):
    axBM = chdr.get_axis(1)
    
    axTP = nb.cifti2.SeriesAxis(start=0, step=1, size=numTrs)
    hdr = nb.cifti2.Cifti2Header.from_axes((axTP, axBM))
    return hdr

