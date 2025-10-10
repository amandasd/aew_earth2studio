# African Easterly Waves Tracking

This example show how to run the AEW tracker using a typical **Earth2Studio** workflow.

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

start_time = datetime(2010, 7, 1)  # Start date
nsteps = 10  # Number of steps to run the tracker for into future
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
