# Dock alignment guidance for centering the AUV camera on the funnel dock.
#
# Sign convention (image coords: x right, y down):
#   error_x_px > 0  → dock center is RIGHT of image center → move/yaw right to center
#   error_y_px > 0  → dock center is BELOW image center → dive / pitch down (NED +z)

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
