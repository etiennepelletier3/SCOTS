# Reconstruct a wavefront from SHS data

## Initiating the reconstructor object

``` python
from lri_ao.reconstructors import SHS_reconstructor

Reconstructor = SHS_reconstructor(
    microlenses_number,
    microlenses_focal_length,
    microlenses_pitch,
    image_resolution=128,
    reconstruction_radius=1,
)
```


## Getting a center of gravity measurement from an image

``` python
Reconstructor.cog_measurement(img)
```


## Use an automatic reconstructor object

``` python

from lri_ao.reconstructors import SHS_AUTO_INIT
Reconstructor = SHS_AUTO_INIT(img, microlenses_focal=10, thresh=0.35)
gy = Reconstructor.cog_measurement(img)[1]
gx = Reconstructor.cog_measurement(img)[0]
```


## Use the zonal reconstructor on an image

``` python
from lri_ao.reconstructors import SHS_AUTO_INIT
from PIL import Image
from pathlib import Path

with Image.open((Path(__file__).parents[0]) / "../media/OL-Test0.tiff") as im:
    a = np.asarray(im)

recon = SHS_AUTO_INIT(a, microlenses_focal=10, thresh=0.15)
recon.mask = recon.r <= 1

# get 2d map of phase
recon.reconstruction_zonal(a)
# get 1d vector of zernike (2 .. mode_max)
recon.reconstruction_modal(a)
```
