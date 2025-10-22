# African Easterly Waves Tracking

This example shows how to run the AEW tracker using **Earth2Studio**.

## Getting started

1. Clone this repository
1. Install [astral uv](https://github.com/astral-sh/uv) if you have not done so previously
1. Update the `uv` environment for this directory: `uv sync`, which will also install `earth2studio`
1. Compile the helper code
```
cc -shared -Wl,-soname,C_circle_functions -o C_circle_functions.so -fPIC C_circle_functions.c
```

## ARCO data source

```python
import torch
import numpy as np
from datetime import datetime, timedelta 
from earth2studio.data import ARCO
from earth2studio.models.dx import aews_detect
from earth2studio.data import prep_data_array

# Create the data source
data = ARCO()

device = torch.device("cpu")

# Create AEW tracker
tracker = aews_detect()
tracker = tracker.to(device)
tracker.detect._device = device
tracker.detect.reset_path_buffer()

start_time = datetime(2010, 9, 23)  # Start date
nsteps = 50  # Number of steps to run the tracker for into future
times = [start_time + timedelta(hours=6 * i) for i in range(nsteps+1)]

for step, time in enumerate(times):
    da = data(time, tracker.detect.input_coords()["variable"])
    x, coords = prep_data_array(da, device=device)
    tracker.detect._current_time = np.array([time])
    if step < nsteps:
       tracker.detect._next_time = np.array([times[step+1]])
    else:
       tracker.detect._next_time = None
    output, output_coords = tracker.detect(x, coords)

out, out_coords = tracker.filter(output, output_coords)
```

## Plotting the tracks

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

aew_tracks = out.cpu()
paths = aew_tracks.numpy()

fig,ax = plt.subplots(ncols=1, figsize=(15,10), subplot_kw={'projection':ccrs.PlateCarree()})
ax.tick_params(axis='both', which='major', labelsize=14)

ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=0.8)
ax.add_feature(cfeature.COASTLINE, linestyle='-', edgecolor='black', linewidth=0.8)

# Add gridlines with labels
gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, color='gray')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size':10, 'color':'black'}
gl.ylabel_style = {'size':10, 'color':'black'}

for path in range(paths.shape[1]):
    # Get lat/lon coordinates, filtering out nans
    lats = paths[0,path,:,5]
    lons = paths[0,path,:,6]
    mask = ~np.isnan(lats) & ~np.isnan(lons)
    if mask.any() and len(lons[mask]) > 2:
        ax.scatter(lons[mask], lats[mask], marker="o", s=10)

fig.savefig('earth2studio_aew_tracks.png', dpi=300, facecolor='w', edgecolor='w', orientation='portrait', bbox_inches='tight')
plt.close(fig)
```
## AEW Tracks — September 1 to October 31, 2010
<img width="3673" height="1008" alt="earth2studio_aew_tracks" src="https://github.com/user-attachments/assets/241219f5-bce7-40fc-9183-86fce2b2f5ae" />

