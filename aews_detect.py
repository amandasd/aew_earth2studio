from collections import OrderedDict

from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem

import numpy
import torch
if torch.cuda.is_available():
    import cupy
np = numpy

from .Tracking_functions import *

# Common info class.
# Each instance/object of Common_track_data is an object that holds
# the latitude and longitude arrays, the lat and lon indices for the
# boundaries over Africa/the Atlantic, the time step for the data,
# the min_threshold value, and the radius.
class Common_track_data:
   def __init__(self):
      self.lat = None
      self.lon = None
      self.dlat = None
      self.dlon = None
      self.lat_index_north = None
      self.lat_index_south = None
      self.lon_index_east = None
      self.lon_index_west = None
      self.lat_index_north_crop = None
      self.lat_index_south_crop = None
      self.lon_index_east_crop = None
      self.lon_index_west_crop = None
      self.total_lon_degrees = None
      self.dt = None
      self.min_threshold = None
      self.radius = None
      self.device = None

class Tracking(torch.nn.Module):

   def __init__(self, config: Common_track_data) -> None:
      super().__init__()
      self.config = config
      self.register_buffer("path_buffer", torch.empty(0))

      # box of interest over Africa/the Atlantic (values are from Albany)
      self.north_lat = 30. #35.
      self.south_lat = 5.
      self.west_lon = -45.
      self.east_lon = 25.
      # lat/lon values to crop data to speed up vorticity calculations
      self.north_lat_crop = 50. #35.
      self.south_lat_crop = -20.
      self.west_lon_crop = -80.
      self.east_lon_crop = 40.

   def reset_path_buffer(self) -> None:
      """Resets the internal"""
      self.path_buffer = torch.empty(0)

   def input_coords(self) -> CoordSystem:
      """Input coordinate system of the diagnostic model

      Returns
      -------
      CoordSystem
          Coordinate system dictionary
      """
      return OrderedDict(
          {
              "batch": np.empty(0),
              "variable": np.array(["u850", "u700", "u600", "v850", "v700", "v600"]),
              "lat": np.linspace(90, -90, 721, endpoint=True),
              "lon": np.linspace(0, 360, 1440, endpoint=False),
          }
      )

   @batch_coords()
   def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
      """Output coordinate system of the diagnostic model

      Parameters
      ----------
      input_coords : CoordSystem
          Input coordinate system to transform into output_coords

      Returns
      -------
      CoordSystem
          Coordinate system dictionary
      """
      target_input_coords = self.input_coords()
      handshake_dim(input_coords, "lon", 3)
      handshake_dim(input_coords, "lat", 2)
      handshake_dim(input_coords, "variable", 1)
      handshake_coords(input_coords, target_input_coords, "lon")
      handshake_coords(input_coords, target_input_coords, "lat")
      handshake_coords(input_coords, target_input_coords, "variable")

      # The lat, lon and magnitude show the progression of the AEW in time,
      # so the first lat/lon point is where the AEW starts, and then
      # the following lat/lon points are where it travels to over time.
      return OrderedDict(
          [
              ("batch", input_coords["batch"]),
              ("path_id", np.empty(0)),
              ("step", np.empty(0)),
              ("variable", np.array(["year", "month", "day", "hour", "minute", "lat", "lon", "mag", "status"])),
          ]
      )

   def initialize_from_dataset(self):

      # get lat and lon values
      # originally lat array goes from north to south
      # (so 90, 89, 88, .....-88, -89, -90)
      # flip makes the lat array go from south to north
      t_lat = torch.as_tensor(self.input_coords()["lat"], device=self.config.device, dtype=torch.float32).flip(0)
      # originally lon array goes from 0-360 degress
      # make the longitude go from -180 to 180 degrees
      t_lon = torch.as_tensor(self.input_coords()["lon"], device=self.config.device, dtype=torch.float32) - 180.0

      # delta lon and delta lat
      self.config.dlon = torch.deg2rad(torch.abs(t_lon[2] - t_lon[1]))
      self.config.dlat = torch.deg2rad(torch.abs(t_lat[2] - t_lat[1]))

      # get north, south, east, west indices for cropping
      self.config.lat_index_north_crop = torch.argmin(torch.abs(t_lat - self.north_lat_crop)).item()
      self.config.lat_index_south_crop = torch.argmin(torch.abs(t_lat - self.south_lat_crop)).item()
      self.config.lon_index_west_crop = torch.argmin(torch.abs(t_lon - self.west_lon_crop)).item()
      self.config.lon_index_east_crop = torch.argmin(torch.abs(t_lon - self.east_lon_crop)).item()

      # crop the lat and lon arrays. We don't need the entire global dataset
      t_lat_crop = t_lat[self.config.lat_index_south_crop:self.config.lat_index_north_crop+1]
      t_lon_crop = t_lon[self.config.lon_index_west_crop:self.config.lon_index_east_crop+1]

      # get north, south, east, west indices for tracking
      self.config.lat_index_north = torch.argmin(torch.abs(t_lat_crop - self.north_lat)).item()
      self.config.lat_index_south = torch.argmin(torch.abs(t_lat_crop - self.south_lat)).item()
      self.config.lon_index_west = torch.argmin(torch.abs(t_lon_crop - self.west_lon)).item()
      self.config.lon_index_east = torch.argmin(torch.abs(t_lon_crop - self.east_lon)).item()

      # Convert a pytorch tensor to either a cupy or numpy array,
      # depending on which device
      if not torch.cuda.is_available() or str(self.config.device) == "cpu":
          np = numpy
          lat_crop = np.asarray(t_lat_crop.detach().cpu().numpy())
          lon_crop = np.asarray(t_lon_crop.detach().cpu().numpy())
      else:
          np = cupy
          lat_crop = cupy.asarray(t_lat_crop.detach())
          lon_crop = cupy.asarray(t_lon_crop.detach())
      # make the lat and lon arrays from the GCM 2D (ordered lat, lon)
      lon = np.tile(lon_crop, (lat_crop.shape[0], 1))
      lat_2d = np.tile(lat_crop, (len(lon_crop), 1))
      lat = np.rot90(lat_2d, 3)
      # make lat and lon arrays C contiguous
      self.config.lat = np.ascontiguousarray(lat, dtype=np.float32)
      self.config.lon = np.ascontiguousarray(lon, dtype=np.float32)

      # the total number of degrees in the longitude dimension
      self.config.total_lon_degrees = np.abs(lon[0,0] - lon[0,-1])

      self.config.initialized = True

   def get_variable(self, x: torch.Tensor, var: str) -> torch.Tensor:
      index = ["u850", "u700", "u600", "v850", "v700", "v600", ].index(var)
      # originally data goes from north to south
      # (so 90, 89, 88, .....-88, -89, -90)
      # flip makes the data go from south to north
      var_raw = x[index].flip(0)
      # originally data goes from 0-360 degress
      # roll makes the data go from -180 to 180 degrees
      var_shifted = torch.roll(var_raw, shifts=int(var_raw.shape[1]/2), dims=1)
      # Crop the data. We don't need the entire global dataset
      var_crop = var_shifted[
            self.config.lat_index_south_crop:self.config.lat_index_north_crop+1,
            self.config.lon_index_west_crop:self.config.lon_index_east_crop+1]
      return var_crop

   # Calculate the relative vorticity: dv/dx - du/dy
   def calc_rel_vort(
      self,
      u: torch.Tensor,
      v: torch.Tensor,
    ) -> torch.Tensor:
      # convert static lat to tensor on same device
      lat = torch.as_tensor(self.config.lat[:, 0], device=self.config.device)

      # calculate dx by multiplying dlon by the radius of the Earth, 6367500m,
      # and the cos of the lat
      dx = 6367500.0 * torch.cos(torch.deg2rad(lat)) * self.config.dlon
      # calculate dy by multiplying dlat by the radius of the Earth, 6367500m
      dy = 6367500.0 * self.config.dlat

      # central difference for dv/dx
      dv_dx = (torch.roll(v, -1, dims=1) - torch.roll(v, 1, dims=1)) / (2 * dx[:, None])
      # fix boundaries: forward/backward
      dv_dx[:, 0] = (v[:, 1] - v[:, 0]) / dx
      dv_dx[:, -1] = (v[:, -1] - v[:, -2]) / dx

      # central difference for du/dy
      du_dy = (torch.roll(u, -1, dims=0) - torch.roll(u, 1, dims=0)) / (2 * dy)
      # fix boundaries: forward/backward
      du_dy[0, :] = (u[1, :] - u[0, :]) / dy
      du_dy[-1, :] = (u[-1, :] - u[-2, :]) / dy

      # relative vorticity
      rel_vort = dv_dx - du_dy

      # switch array to float32 instead of float64
      # make array C contiguous
      return rel_vort.to(dtype=torch.float32, memory_format=torch.contiguous_format)

   # Calculate the curvature vorticity
   def calc_curve_vort(
      self,
      u: torch.Tensor,
      v: torch.Tensor,
      rel_vort: torch.Tensor,
   ) -> torch.Tensor:
      # convert static lat to tensor on same device
      lat = torch.as_tensor(self.config.lat[:, 0], device=self.config.device)

      # calculate dx by multiplying dlat by the radius of the Earth, 6367500m,
      # and the cos of the lat
      #TODO: dlat?
      dx = 6367500.0 * torch.cos(torch.deg2rad(lat)) * self.config.dlon
      # calculate dy by multiplying dlat by the radius of the Earth, 6367500m
      dy = 6367500.0 * self.config.dlat

      # calculate the magnitude of the wind vector sqrt(u^2+v^2),
      # and then make u and v unit vectors
      vec_mag = torch.sqrt(torch.pow(u, 2) + torch.pow(v, 2))
      u_unit_vec = u / vec_mag
      v_unit_vec = v / vec_mag

      # previous longitude
      u_prev_lon = torch.roll(u, shifts=1, dims=1)
      v_prev_lon = torch.roll(v, shifts=1, dims=1)
      u_prev_lon[:, 0] = u[:, 0] # clamp edge
      v_prev_lon[:, 0] = v[:, 0]

      # next longitude
      u_next_lon = torch.roll(u, shifts=-1, dims=1)
      v_next_lon = torch.roll(v, shifts=-1, dims=1)
      u_next_lon[:, -1] = u[:, -1] # clamp edge
      v_next_lon[:, -1] = v[:, -1]

      # previous latitude
      u_prev_lat = torch.roll(u, shifts=1, dims=0)
      v_prev_lat = torch.roll(v, shifts=1, dims=0)
      u_prev_lat[0, :] = u[0, :] # clamp edge
      v_prev_lat[0, :] = v[0, :]

      # next latitude
      u_next_lat = torch.roll(u, shifts=-1, dims=0)
      v_next_lat = torch.roll(v, shifts=-1, dims=0)
      u_next_lat[-1, :] = u[-1, :] # clamp edge
      v_next_lat[-1, :] = v[-1, :]

      # calculate v1, v2, u1, and u2
      v1 = (u_unit_vec * u_prev_lon + v_unit_vec * v_prev_lon) * v_unit_vec
      v2 = (u_unit_vec * u_next_lon + v_unit_vec * v_next_lon) * v_unit_vec
      u1 = (u_unit_vec * u_prev_lat + v_unit_vec * v_prev_lat) * u_unit_vec
      u2 = (u_unit_vec * u_next_lat + v_unit_vec * v_next_lat) * u_unit_vec

      # di counts valid longitude neighbors
      di = torch.ones_like(u, dtype=torch.float32) * 2 # default: both neighbors exist
      di[:, 0] = 1 # at lon=0, only "next" neighbor valid
      di[:, -1] = 1 # at lon=max, only "prev" neighbor valid

      # dj counts valid latitude neighbors
      dj = torch.ones_like(u, dtype=torch.float32) * 2
      dj[0, :] = 1 # at lat=0, only "next" neighbor valid
      dj[-1, :] = 1 # at lat=max, only "prev" neighbor valid

      shear_vort = ((v2 - v1) / (di * dx[:, None])) - ((u2 - u1) / (dj * dy))
      curve_vort = rel_vort - shear_vort
      # switch array to float32 instead of float64
      # make array C contiguous
      curve_vort = curve_vort.to(dtype=torch.float32).contiguous()
      return curve_vort

   @batch_func()
   def __call__(
      self,
      x: torch.Tensor,
      coords: CoordSystem,
   ) -> tuple[torch.Tensor, CoordSystem]:
      """Runs diagnostic model

      Parameters
      ----------
      x : torch.Tensor
          Input tensor
      coords : CoordSystem
          Input coordinate system
      """
      # get the device to run on
      if not torch.cuda.is_available() or str(self._device) == "cpu":
          np = numpy
          use_gpu = False
          self.config.device = torch.device("cpu")
      else:
          np = cupy
          use_gpu = True
          self.config.device = torch.device(self._device)

      if ~self.config.initialized:
         self.initialize_from_dataset()

      outs = []
      for i in range(x.shape[0]):
          # get u and v on the levels 850, 700, and 600 hP for a given time step
          t_u850 = self.get_variable(x[i], "u850")
          t_u700 = self.get_variable(x[i], "u700")
          t_u600 = self.get_variable(x[i], "u600")
          t_v850 = self.get_variable(x[i], "v850")
          t_v700 = self.get_variable(x[i], "v700")
          t_v600 = self.get_variable(x[i], "v600")

          # calculate the relative vorticity
          t_rel_vort_850 = self.calc_rel_vort(t_u850, t_v850)
          t_rel_vort_700 = self.calc_rel_vort(t_u700, t_v700)
          t_rel_vort_600 = self.calc_rel_vort(t_u600, t_v600)
          t_rel_vort = torch.stack([t_rel_vort_850, t_rel_vort_700, t_rel_vort_600], dim=0)
          # calculate the curvature vorticity
          t_curve_vort_850 = self.calc_curve_vort(t_u850, t_v850, t_rel_vort_850)
          t_curve_vort_700 = self.calc_curve_vort(t_u700, t_v700, t_rel_vort_700)
          t_curve_vort_600 = self.calc_curve_vort(t_u600, t_v600, t_rel_vort_600)
          t_curve_vort = torch.stack([t_curve_vort_850, t_curve_vort_700, t_curve_vort_600], dim=0)

          # Convert a pytorch tensor to either a cupy or numpy array,
          # depending on which device
          if use_gpu:
              rel_vort = cupy.asarray(t_rel_vort.detach())
              curve_vort = cupy.asarray(t_curve_vort.detach())
              u850 = cupy.asarray(t_u850.detach())
              v850 = cupy.asarray(t_v850.detach())
              u600 = cupy.asarray(t_u600.detach())
              v600 = cupy.asarray(t_v600.detach())
          else:
              rel_vort = t_rel_vort.detach().cpu().numpy()
              curve_vort = t_curve_vort.detach().cpu().numpy()
              u850 = t_u850.detach().cpu().numpy()
              v850 = t_v850.detach().cpu().numpy()
              u600 = t_u600.detach().cpu().numpy()
              v600 = t_v600.detach().cpu().numpy()

          # smooth the curvature and relative vorticities
          curve_vort_smooth = c_smooth(self.config, curve_vort, self.config.radius*1.5)
          rel_vort_smooth = c_smooth(self.config, rel_vort, self.config.radius*1.5)
          # Convert a cupy or numpy array to a pytorch tensor
          t_curve_vort_smooth = torch.tensor(curve_vort_smooth, dtype=torch.float32, device=str(self._device))
          t_rel_vort_smooth = torch.tensor(rel_vort_smooth, dtype=torch.float32, device=str(self._device))

          # Find new starting points
          # This function takes the curvature vorticity at 700 hPa
          unique_max_locs = get_starting_targets(self.config, curve_vort_smooth[1])

          # Combine potential locations into tracked locations
          # This function takes the curvature vorticity at 850 and 600 hPa
          # and the relative vorticity at 700 and 600 hPa.
          # Curvature vorticity at 700 hPa was already looked at in get_starting_targets.
          # Relative vorticity at 850 hPat was not included in the Albany program.
          #TODO: radius is 350km in unique_locations function?
          alternative_unique_max_locs = get_multi_positions(self.config, curve_vort_smooth, rel_vort_smooth, unique_max_locs)

          np = numpy

          # Remove duplicate locations
          # The 99999999 is the starting value for unique_loc_number;
          # it just needs to be way bigger than the possible number of
          # local maxima in weighted_max_indices.
          # This function recursively calls itself until the value for
          # unique_loc_number doesn't decrease anymore.
          # unique_max_locs+alternative_unique_max_locs is joining the two lists
          # (e.g. [a,b,c]+[d,e,f] -> [a,b,c,d,e,f]).
          # Use the following conditional to catch the case where there is
          # only one location, in which case we don't need to use unique_locations.
          if len(unique_max_locs + alternative_unique_max_locs) > 1:
             combined_unique_max_locs = unique_locations("cpu", unique_max_locs + alternative_unique_max_locs, self.config.radius, 99999999)
          else:
             combined_unique_max_locs = unique_max_locs + alternative_unique_max_locs

          # Compare combined_unique_max_locs at a given time with the
          # track locations at the same time.
          # If there are locations that are too close together
          # (so duplicates between the existing track and the new track locations
          # that are in combined_unique_max_locs), average the two lat/lon pairs,
          # then use this new value to replace the old lat/lon value in the
          # existing track, and remove the duplicate lat/lon location from
          # combined_unique_max_locs.
          # only enter if path_buffer isn't empty
          if self.path_buffer.numel() != 0 and combined_unique_max_locs:
             for track in range(self.path_buffer[i].size(0)):
                 # check the tracking status
                 # (false means finished track because of weak magnitude, i.e.
                 # inactive track)
                 if self.path_buffer[i,track,-1,-1]:
                    # get the last lat/lon location
                    track_lat = self.path_buffer[i,track,-1,5]
                    track_lon = self.path_buffer[i,track,-1,6]
                    if combined_unique_max_locs:
                       # check to make sure that the new track locations aren't
                       # duplicates of existing tracks
                       #TODO: check unique_track_locations
                       new_latlon_pair = unique_track_locations("cpu", (track_lat,track_lon), combined_unique_max_locs, self.config.radius)
                       self.path_buffer[i,track,-1,5] = new_latlon_pair[0]
                       self.path_buffer[i,track,-1,6] = new_latlon_pair[1]
                    else:
                       break

          # for the locations in combined_unique_max_locs
          # (assuming it isn't empty), create new AEW tracks.
          new_tracks = []
          if combined_unique_max_locs:
             for lat_lon_pair in combined_unique_max_locs:
                 new_tracks.append([
                  self._current_time.item().year, self._current_time.item().month,
                  self._current_time.item().day, self._current_time.item().hour,
                  self._current_time.item().minute,
                  lat_lon_pair[0], lat_lon_pair[1], np.nan, True])

          tracks_list = []
          if self.path_buffer.numel() != 0:
             if new_tracks:
                new_tracks_reshaped = torch.tensor(new_tracks, dtype=torch.float32).unsqueeze(1)
                new_tracks_padded = torch.cat([torch.full((len(new_tracks), self.path_buffer[i].size(1)-1, 9), float('nan')), new_tracks_reshaped], dim=1)
                # Append the new and old tracks to tracks_list.
                # The tracks from the previous times are in self.path_buffer
                tracks_list = torch.cat((self.path_buffer[i], new_tracks_padded.detach().cpu()), dim=0)
             else:
                tracks_list = self.path_buffer[i]
          else:
             if new_tracks:
                new_tracks_reshaped = torch.tensor(new_tracks, dtype=torch.float32).unsqueeze(1)
                tracks_list = new_tracks_reshaped
                 
          # loop through all tracks and assign magnitudes to the new lat/lon pairs
          # that have been added to each track.
          # Then filter out any tracks that don't meet AEW qualifications and
          # advect the tracks that are leftover that didn't get filtered.
          next_step = []
          tracks_to_remove = []
          for track in range(tracks_list.size(0)):
              # check the tracking status
              # (false means finished track because of weak magnitude, i.e.
              # inactive track)
              if tracks_list[track,-1,-1]:
                 # get the last lat/lon location
                 track_lat = tracks_list[track,-1,5].item()
                 track_lon = tracks_list[track,-1,6].item()
                 if torch.isnan(tracks_list[track,-1,7]):
                    # Assign magnitude from the vorticity to each lat/lon point.
                    # The magnitude is whatever is largest among
                    # curvature vorticity at 850, 700 and 600 hPa and
                    # relative voriticy at 700 and 600 hPa
                    tracks_list[track,-1,7] = assign_magnitude(self.config, t_curve_vort_smooth, t_rel_vort_smooth, (track_lat,track_lon))
                 # Keep only time steps where lat is not nan
                 valid_idx = torch.where(~torch.isnan(tracks_list[track,:,5]))[0]
                 # Filter tracks. A track is either removed entirely or
                 # the tracking status is updated to an inactive track.
                 filter_result = filter_tracks(self.config, tracks_list[track,valid_idx,5:7], tracks_list[track,valid_idx,7])
                 if filter_result.reject_track_direction:
                    print("in reject track")
                    tracks_to_remove.append(track)
                    del filter_result
                    continue
                 elif filter_result.magnitude_finish_track or filter_result.latitude_finish_track:
                    print("in weak magnitude")
                    # make sure that the next time step is actually within our timeframe
                    if self._next_time is not None:
                       # inactive track
                       next_step.append([np.nan, np.nan, np.nan, np.nan, np.nan,
                                         np.nan, np.nan, np.nan, False])
                    del filter_result
                    continue
                 del filter_result
                 # make sure that the next time step is actually within our timeframe
                 if self._next_time is not None:
                    # advect tracks
                    new_latlon_pair = advect_tracks(self.config, u850, u600, v850, v600, (track_lat,track_lon))
                    next_step.append([self._next_time.item().year,
                                      self._next_time.item().month,
                                      self._next_time.item().day,
                                      self._next_time.item().hour,
                                      self._next_time.item().minute,
                                      new_latlon_pair[0], new_latlon_pair[1],
                                      np.nan, True])
              else:
                 # make sure that the next time step is actually within our timeframe
                 if self._next_time is not None:
                    # inactive track
                    next_step.append([np.nan, np.nan, np.nan, np.nan, np.nan,
                                      np.nan, np.nan, np.nan, False])
          if tracks_to_remove:
             mask = torch.ones(tracks_list.size(0), dtype=torch.bool)
             mask[tracks_to_remove] = False
             tracks_list = tracks_list[mask]
          if next_step:
             next_step_reshaped = torch.tensor(next_step, dtype=torch.float32).unsqueeze(1)
             tracks_list = torch.cat((tracks_list, next_step_reshaped.detach().cpu()), dim=1)
          outs.append(torch.tensor(tracks_list, device="cpu"))

      out = torch.nn.utils.rnn.pad_sequence(outs, padding_value=np.nan, batch_first=True).cpu()

      # Accumulate AEW tracks over time steps into path_buffer
      if self.path_buffer.numel() == 0:
          self.path_buffer = out.detach().clone().cpu()
      else:
          self.path_buffer = out

      # Update out_coords with path_id and step identifiers
      out_coords = self.output_coords(coords)
      out_coords["path_id"] = np.arange(out.shape[1])
      out_coords["step"] = np.arange(out.shape[2])

      return self.path_buffer, out_coords

class Filtering(torch.nn.Module):

   def __init__(self, config: Common_track_data) -> None:
      super().__init__()
      self.config = config

   def input_coords(self) -> CoordSystem:
      """Input coordinate system of the diagnostic model

      Returns
      -------
      CoordSystem
          Coordinate system dictionary
      """
      return OrderedDict(
          [
              ("batch", np.empty(0)),
              ("path_id", np.empty(0)),
              ("step", np.empty(0)),
              ("variable", np.array(["year", "month", "day", "hour", "minute", "lat", "lon", "mag", "status"])),
          ]
      )

   @batch_coords()
   def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
      """Output coordinate system of the diagnostic model

      Parameters
      ----------
      input_coords : CoordSystem
          Input coordinate system to transform into output_coords

      Returns
      -------
      CoordSystem
          Coordinate system dictionary
      """
      return OrderedDict(
          [
              ("batch", input_coords["batch"]),
              ("path_id", np.empty(0)),
              ("step", np.empty(0)),
              ("variable", np.array(["year", "month", "day", "hour", "minute", "lat", "lon", "mag", "status"])),
          ]
      )

   @batch_func()
   def __call__(
      self,
      x: torch.Tensor,
      coords: CoordSystem,
   ) -> tuple[torch.Tensor, CoordSystem]:
      """Filtering to check for tracks that weren't long enough,
         didn't start far enough east, didn't go far enough west, or
         go too far south.

      Parameters
      ----------
      x : torch.Tensor
          Input tensor
      coords : CoordSystem
          Input coordinate system
      """
      np = numpy

      outs = []
      for i in range(x.shape[0]):
          tracks_list = x[i]
          tracks_to_remove = []
          for track in range(tracks_list.size(0)):
              # Keep only time steps where lat is not nan
              valid_idx = torch.where(~torch.isnan(tracks_list[track,:,5]))[0]
              # Check for tracks that haven't lasted long enough.
              # If the track hasn't lasted for two days
              # (which is < 48/dt + 1 time step), get rid of it
              if len(tracks_list[track,valid_idx,5:7]) < ((48/self.config.dt)+1):
                 print("not enough times")
                 tracks_to_remove.append(track)
                 continue
              # Check to see if the distance along the track is less than 15 degress,
              # if it is, get rid of it. To do this, find the differences:
              # the second-last lats minus first-second to last lats and
              # the second-last lons minus first-second to last lons.
              # These differences are the deltas in between the latitudes and
              # the longitudes of the track. Then calculate the distance between
              # each lat/lon point using these deltas and
              # the standart dist = sqrt( (x2-x1)^2 + (y2-y1)^2 ).
              # Then take the sum of these distances. If the sum of the distances
              # between each lat/lon point is < 15 degrees, remove the track.
              if torch.sum(torch.sqrt((tracks_list[track,valid_idx,5][1:] - tracks_list[track,valid_idx,5][:-1])**2
                                    + (tracks_list[track,valid_idx,6][1:] - tracks_list[track,valid_idx,6][:-1])**2)) < 15:
                 print("short track")
                 tracks_to_remove.append(track)
                 continue
              # if the smallest (farthest west) longitude value
              # is greater than -20 (20W), which means east of 20W, then
              # remove the track because it doesn't travel far enough west
              # last valid index
              last_idx = valid_idx[-1].item()
              if tracks_list[track,last_idx,6] > -20:
                 print("not far enough west")
                 tracks_to_remove.append(track)
                 continue
              # if the largest (farthest east) longitude value
              # is less than -5 (5W), which means west of 5W, then
              # remove the track because it doesn't start far enough east
              # first valid index
              first_idx = valid_idx[0].item()
              if tracks_list[track,first_idx,6] < -5:
                 print("doesn't start far enough east")
                 tracks_to_remove.append(track)
                 continue
              # if the largest (farthest north) latitude value
              # is less than 5 (5N), then remove the track
              # because it is too far south
              if torch.max(tracks_list[track,valid_idx,5]) < 5:
                 print("too far south")
                 tracks_to_remove.append(track)
                 continue

          if tracks_to_remove:
             mask = torch.ones(tracks_list.size(0), dtype=torch.bool)
             mask[tracks_to_remove] = False
             tracks_list = tracks_list[mask]
          outs.append(torch.tensor(tracks_list, device="cpu"))

      out = torch.nn.utils.rnn.pad_sequence(outs, padding_value=np.nan, batch_first=True).cpu()

      # Update out_coords with path_id and step identifiers
      out_coords = self.output_coords(coords)
      out_coords["path_id"] = np.arange(out.shape[1])
      out_coords["step"] = np.arange(out.shape[2])

      return out, out_coords

class aews_detect(torch.nn.Module):
   """Custom diagnostic model"""

   def __init__(self):
      super().__init__()
      # create an object from Common_track_data that
      # will hold lat, lon, dt, and other information
      self.config = Common_track_data()
      # 6-hour time step
      self.config.dt = 6
      # set radius in km (ERA5) for smoothing and for finding points
      # that belong to the same track
      self.config.radius = 700.0
      # Brannan and Martin (2019) used 0.0000015
      self.config.min_threshold = 0.000002
      self.config.initialized = False

      self.detect = Tracking(self.config)
      self.filter = Filtering(self.config)


