# Dock alignment guidance for centering the AUV camera on the funnel dock.
#
# Sign convention (image coords: x right, y down):
#   error_x_px > 0  → dock center is RIGHT of image center → yaw right to center
#   error_y_px > 0  → dock center is BELOW image center → pitch down to keep it in frame
#   elevation_rad > 0 → dock center is BELOW the horizon → heave down (NED +z)
#   lateral_px > 0 → side-light midpoint is RIGHT of the top→bottom line.
#                    Near 0 when the heading is perpendicular to the dock, even if
#                    the vehicle is shifted sideways. A sideways shift with a
#                    perpendicular heading shows up in error_x_px instead.
#                    There is no sway thruster: yaw toward error_x and surge.
#   search_* are used only while valid is false (fewer than four trusted lights).
#   search_yaw_norm > 0 → yaw toward image right (needs forward speed for the fins)
#   search_pitch_norm > 0 → pitch down
#   search_surge_norm > 0 → axial thruster forward; < 0 → reverse

std_msgs/Header header

bool valid
uint8 num_lights

float32 error_x_px
float32 error_y_px
float32 error_x_norm
float32 error_y_norm

uint32 image_width
uint32 image_height

float32 radius_px
float32 spread_px
float32 err_left_px
float32 err_right_px
bool aligned
float32 diameter_angle_deg

bool center_exact
float32 lateral_px
float32 obliqueness
float32 elevation_rad
bool elevation_valid

float32 search_yaw_norm
float32 search_pitch_norm
float32 search_surge_norm

geometry_msgs/Point center
geometry_msgs/Point top
geometry_msgs/Point bottom
geometry_msgs/Point left
geometry_msgs/Point right

float32 d_top
float32 d_bottom
float32 d_left
float32 d_right

string status
float32 confidence
