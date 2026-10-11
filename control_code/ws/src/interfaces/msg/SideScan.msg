# One ping of ONE side-scan sonar (simulated Cerulean Omniscan 450 SS; same idea as the Omniscan API's profile data: an intensity per range bin).
# intensity[i] is the echo power in the range bin [start_range_m + i*dr, start_range_m + (i+1)*dr), dr = (range_m - start_range_m) / num_bins, scaled to 0..65535
# (log scale over dynamic_range_db, after time-varied gain). Range is the SLANT range from the transducer (round trip time * sound speed / 2).
std_msgs/Header header
string side               # "port" or "starboard"
uint32 ping_number
float32 start_range_m
float32 range_m           # end of the profile
float32 sound_speed_mps
int8 gain_index           # -1 = auto gain, 0..7 manual
float32 gain_db           # gain actually applied
float32 ping_hz
float32 altitude_m        # altimeter reading under the vehicle at the ping (NaN = none)
float64[3] vehicle_pos_ned     # vehicle pose at transmit (truth of the simulator, for the mosaic; the real sonar does not send this)
float64[3] vehicle_eul_rad     # roll, pitch, yaw
uint32 num_bins
uint16[] intensity
