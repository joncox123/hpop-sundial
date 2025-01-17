"""
--- High-precision cylindrical sundial generator using Astropy ---

MIT License

Copyright (c) 2025, Jonathan A. Cox

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time
import astropy.units as u
from datetime import datetime, timedelta
import pytz
from tqdm import tqdm


# set fig size scaling factor so that figure saves as PDF true to size
# change this for your system
DPI_SCALE = 1.298279779


def get_solar_angle(latitude, longitude, elevation_MSL, datetime_UTC):
    """
    Calculate the apparent angle of the sun in terms of azimuth and elevation.

    Parameters:
    latitude (float): Latitude of the location in degrees.
    longitude (float): Longitude of the location in degrees.
    elevation_MSL (float): Elevation above mean sea level in meters.
    datetime_UTC (str): Date and time in UTC (ISO 8601 format).

    Returns:
    tuple: Azimuth and elevation of the sun in degrees.
    """
    location = EarthLocation(lat=latitude*u.deg, lon=longitude*u.deg, height=elevation_MSL*u.m)
    time = Time(datetime_UTC, format='isot', scale='utc')
    altaz = AltAz(obstime=time, location=location)
    sun = get_sun(time).transform_to(altaz)
    
    return sun.az.deg, sun.alt.deg

def calculate_shadow_position(latitude, R, azimuth, elevation):
    """
    Calculate the shadow position of the gnomon tip on a cylinder-shaped sundial.

    Parameters:
    latitude (float): Latitude of the location in degrees.
    R (float): Radius of the cylinder.
    azimuth (float): Azimuth angle of the sun in degrees.
    elevation (float): Elevation angle of the sun in degrees.

    Returns:
    tuple: (x, y, z) coordinates of the shadow on the cylinder, and the sun direction vector.
    """
    # Convert angles to radians
    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)
    latitude_rad = np.radians(latitude)

    # Calculate the direction of the sun in the local coordinate system
    sun_direction = -np.array([
        np.cos(elevation_rad) * np.sin(azimuth_rad),
        np.cos(elevation_rad) * np.cos(azimuth_rad),
        np.sin(elevation_rad)
    ])
    sun_direction = sun_direction / np.linalg.norm(sun_direction)

    # Use tilt_angle = (π/2) − latitude_rad so cylinder is vertical at lat=90 and horizontal at lat=0
    tilt_angle = (np.pi / 2) - latitude_rad
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(tilt_angle), -np.sin(tilt_angle)],
        [0, np.sin(tilt_angle),  np.cos(tilt_angle)]
    ])
    sun_direction_rotated = np.dot(rotation_matrix_x, sun_direction)
    sun_direction_rotated = sun_direction_rotated / np.linalg.norm(sun_direction_rotated)

    # Intersection in rotated frame
    t = R / np.sqrt(sun_direction_rotated[0]**2 + sun_direction_rotated[1]**2)
    x_shadow = t * sun_direction_rotated[0]
    y_shadow = t * sun_direction_rotated[1]
    z_shadow = t * sun_direction_rotated[2]

    # Convert back to true frame
    inverse_rotation_matrix_x = np.linalg.inv(rotation_matrix_x)
    shadow_position = np.dot(inverse_rotation_matrix_x, np.array([x_shadow, y_shadow, z_shadow]))

    return shadow_position[0], shadow_position[1], shadow_position[2], sun_direction


def project_3D_to_paper(sun_path_positions, R, R_pipe, latitude):
    """
    Project the 3D sun path positions from a cylinder of radius R onto a 2D paper
    that can be rolled into a cylinder of radius R_pipe.

    Parameters:
    sun_path_positions (list of tuples): List of (x, y, z) coordinates of the sun path positions.
    R (float): Radius of the original cylinder.
    R_pipe (float): Radius of the paper cylinder.
    latitude (float): Latitude of the location in degrees.

    Returns:
    tuple: x_values and y_values for plotting on paper coordinates.
    """
    # Calculate the tilt angle of the cylinder based on the latitude
    tilt_angle = (np.pi / 2) - np.radians(latitude)
    
    # Rotation matrix to align the cylinder's axis with the vertical axis
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(tilt_angle), -np.sin(tilt_angle)],
        [0, np.sin(tilt_angle),  np.cos(tilt_angle)]
    ])

    x_values = []
    y_values = []

    for (x, y, z) in sun_path_positions:
        # Rotate the (x, y, z) point to the cylinder's upright frame
        x_rot, y_rot, z_rot = rotation_matrix_x.dot([x, y, z])
        
        # Calculate the angle theta in the xy-plane
        theta = -np.arctan2(y_rot, x_rot)
        
        # Project the angle theta onto the paper's x-axis
        x_proj = R_pipe * theta
        
        # Scale the z-coordinate to the paper's y-axis
        scale = R_pipe / R
        x_values.append(x_proj)
        y_values.append(z_rot * scale)

    x_values = np.array(x_values)
    y_values = np.array(y_values)

    return x_values, y_values


def plot_cylinder_with_shadows(latitude, R, shadow_positions, sun_vectors):
    """
    Plot the wireframe of the cylinder in 3D along with the shadow positions and sun vectors.

    Parameters:
    latitude (float): Latitude of the location in degrees.
    R (float): Radius of the cylinder.
    shadow_positions (list of tuples): List of (x, y, z) coordinates of the shadow on the cylinder.
    sun_vectors (list of tuples): List of sun direction vectors.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create the cylinder
    z = np.linspace(-1.5*R, 1.5*R, 50)
    theta = np.linspace(-np.pi, 0, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = R * np.cos(theta_grid)
    y_grid = R * np.sin(theta_grid)

    # Rotate the cylinder to match the tilt based on the latitude
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(np.radians(latitude)+np.pi/2), -np.sin(np.radians(latitude)+np.pi/2)],
        [0, np.sin(np.radians(latitude)+np.pi/2), np.cos(np.radians(latitude)+np.pi/2)]
    ])
    x_grid_rotated, y_grid_rotated, z_grid_rotated = np.dot(rotation_matrix_x, np.array([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()]))
    x_grid_rotated = x_grid_rotated.reshape(x_grid.shape)
    y_grid_rotated = y_grid_rotated.reshape(y_grid.shape)
    z_grid_rotated = z_grid_rotated.reshape(z_grid.shape)

    # Plot the cylinder wireframe
    ax.plot_wireframe(x_grid_rotated, y_grid_rotated, z_grid_rotated, color='b', alpha=0.2)

    # Plot the shadow positions
    shadow_x, shadow_y, shadow_z = zip(*shadow_positions)
    ax.scatter(shadow_x, shadow_y, shadow_z, color='r', s=12)

    # Plot the gnomon tip
    ax.scatter(0, 0, 0, color='k', s=100, label='Gnomon Tip')

    # Plot the sun vectors
    for vector in sun_vectors:
        ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color='y', length=R, normalize=True)

    # Set labels
    ax.set_xlabel('X (East-West)')
    ax.set_ylabel('Y (North)')
    ax.set_zlabel('Z (Up)')
    ax.set_title('Cylinder with Shadow Positions')
    ax.legend()

    # Ensure the axes are equal scale
    max_range = np.array([x_grid_rotated.max()-x_grid_rotated.min(), y_grid_rotated.max()-y_grid_rotated.min(), z_grid_rotated.max()-z_grid_rotated.min()]).max() / 2.0
    mid_x = (x_grid_rotated.max()+x_grid_rotated.min()) * 0.5
    mid_y = (y_grid_rotated.max()+y_grid_rotated.min()) * 0.5
    mid_z = (z_grid_rotated.max()+z_grid_rotated.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1


def plot_cylinder_with_shadows_by_hour(latitude, R, sun_path_by_hour_dict):
    """
    Plot the wireframe of the cylinder in 3D along with the shadow positions color-coded by hour.

    Parameters:
    latitude (float): Latitude of the location in degrees.
    R (float): Radius of the cylinder.
    sun_path_by_hour_dict (dict): Dictionary containing sun path positions and vectors.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create the cylinder
    z = np.linspace(-1.5*R, 1.5*R, 50)
    theta = np.linspace(-np.pi, 0, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = R * np.cos(theta_grid)
    y_grid = R * np.sin(theta_grid)

    # Rotate the cylinder to match the tilt based on the latitude
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(np.radians(latitude)+np.pi/2), -np.sin(np.radians(latitude)+np.pi/2)],
        [0, np.sin(np.radians(latitude)+np.pi/2), np.cos(np.radians(latitude)+np.pi/2)]
    ])
    x_grid_rotated, y_grid_rotated, z_grid_rotated = np.dot(rotation_matrix_x, np.array([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()]))
    x_grid_rotated = x_grid_rotated.reshape(x_grid.shape)
    y_grid_rotated = y_grid_rotated.reshape(y_grid.shape)
    z_grid_rotated = z_grid_rotated.reshape(z_grid.shape)

    # Plot the cylinder wireframe
    ax.plot_wireframe(x_grid_rotated, y_grid_rotated, z_grid_rotated, color='b', alpha=0.2)

    # Plot the shadow positions color-coded by hour
    for i, (local_hour, data) in enumerate(sun_path_by_hour_dict.items()):
        sun_path_positions = data['sun_path_positions']
        x_values, y_values, z_values = zip(*sun_path_positions)

        ax.plot(x_values, y_values, z_values, label=f'{local_hour}:00')

        # Annotate the first, middle, and last points with the corresponding day
        if i == 0:
            dt_list = data['dt_list']
            for idx in [0, len(dt_list) // 2, len(dt_list) - 1]:
                day_label = dt_list[idx].strftime('%b %d')
                ax.text(x_values[idx], y_values[idx], z_values[idx], day_label, color='red')

    # Plot the gnomon tip
    ax.scatter(0, 0, 0, color='k', s=100, label='Gnomon Tip')

    # Plot gnomon center
    sx, sy, sz, sv = calculate_shadow_position(latitude, R, 180, 90-latitude)
    ax.scatter(sx, sy, sz, color='k', s=50, label='Gnomon Center')

    # Set labels
    ax.set_xlabel('X (East-West)')
    ax.set_ylabel('Y (North)')
    ax.set_zlabel('Z (Up)')
    ax.set_title('Cylinder with Shadow Positions by Hour')
    ax.legend()

    # Ensure the axes are equal scale
    max_range = np.array([x_grid_rotated.max()-x_grid_rotated.min(), y_grid_rotated.max()-y_grid_rotated.min(), z_grid_rotated.max()-z_grid_rotated.min()]).max() / 2.0
    mid_x = (x_grid_rotated.max()+x_grid_rotated.min()) * 0.5
    mid_y = (y_grid_rotated.max()+y_grid_rotated.min()) * 0.5
    mid_z = (z_grid_rotated.max()+z_grid_rotated.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1


def plot_sundial_on_paper(W, H, R, pipe_diameter, sun_path_by_hour_dict, latitude, longitude, local_tz, output_pdf='sundial.pdf'):
    """
    Plot the sundial paths on a paper of dimensions WxH and save as a PDF.

    Parameters:
    W (float): Width of the paper.
    H (float): Height of the paper.
    R (float): Radius of the cylinder.
    pipe_diameter (float): project the points such that if the paper is
                        rolled into a tube of radius pipe_diameter/2, the dial will be correct.
                        This is useful when pasting the paper to the inside of a half-pipe.
    sun_path_by_hour_dict (dict): Dictionary containing sun path positions and vectors.
    latitude (float): Latitude of the location.
    longitude (float): Longitude of the location.
    local_tz (str): Local time zone.
    output_pdf (str): Output PDF file name.
    
    Returns:
    float: Radius for which the paper must be curled.
    """
    
    R_pipe = pipe_diameter/2

    fig, ax = plt.subplots(figsize=(W*DPI_SCALE, H*DPI_SCALE), dpi=96)

    # Calculate the gnomon position
    sx, sy, sz, sv = calculate_shadow_position(latitude, R, 180, 90-latitude)
    x_gnomon, y_gnomon = project_3D_to_paper([[sx, sy, sz]], R, R_pipe, latitude)
    
    W_os = W/2-x_gnomon
    H_os = H/2

    ax.plot(x_gnomon+W_os, y_gnomon+H_os, 'ko', label='Gnomon Center')

    # Extract unique dates from the sun_path_by_hour_dict
    dates = sorted({dt.date() for data in sun_path_by_hour_dict.values() for dt in data['dt_list']})

    for i, (local_hour, data) in enumerate(sun_path_by_hour_dict.items()):
        sun_path_positions = data['sun_path_positions']
        dt_list = data['dt_list']

        x_values, y_values = project_3D_to_paper(sun_path_positions, R, R_pipe, latitude)
        x_values += W_os
        y_values += H_os

        ax.plot(x_values, y_values, label=f'Hour {local_hour}')
        ax.annotate(f'{local_hour:02d}:00', xy=(x_values[0], y_values[0]), xytext=(-10, 5),
                    textcoords='offset points', color='black', fontsize=12, rotation=45)
        ax.annotate(f'{local_hour:02d}:00', xy=(x_values[-1], y_values[-1]), xytext=(-10, -30),
                    textcoords='offset points', color='black', fontsize=12, rotation=45)

        # # Annotate the first, middle, and last points with the corresponding day
        # if i == 0:
        #     for idx in [0, len(dt_list) // 2, len(dt_list) - 1]:
        #         day_label = dt_list[idx].astimezone(local_tz).strftime('%b %d')
        #         ax.annotate(day_label, xy=(x_values[idx], y_values[idx]), xytext=(5, 5),
        #                     textcoords='offset points', color='red', fontsize=8)
        #         ax.plot(x_values[idx], y_values[idx], 'rx')  # Plot an 'X' marker

    # Draw horizontal lines corresponding to the first of each month
    for i in range(len(y_values)):
        if dt_list[i].day == 1:
            ax.plot([0, W], [y_values[i], y_values[i]], 'k-', linewidth=1.25)
            ax.annotate(dt_list[i].strftime('%b'), xy=(1, y_values[i]), xytext=(0, .2),
                        textcoords='offset points', fontsize=14)
        elif dt_list[i].day % 7 == 0:
            ax.plot([0, W], [y_values[i], y_values[i]], 'gray', linestyle='--', linewidth=1)

    # Plot vertical and horizontal red lines crossing through (0,0) that are exactly 1 inch long
    ax.quiver(W/2, 1, 0, 1, color='r', scale=1, scale_units='xy', angles='xy', width=0.005)  # Vertical arrow
    ax.annotate('North', xy=(W/2, 2), xytext=(0, 5), textcoords='offset points', ha='center', color='red', fontsize=12)
    ax.plot([W/2-0.5, W/2+0.5], [1.5, 1.5], 'k-', linewidth=1)  # Horizontal line

    tilt_angle_deg = np.degrees(np.pi / 2 - np.radians(latitude))
    ax.set_title(f'Sundial Paths on Paper\nLatitude: {latitude:.3f}, Longitude: {longitude:.3f}, Time Zone: {local_tz}\nDiameter: {pipe_diameter:.2f} inches, Tilt Angle: {tilt_angle_deg:.2f} degrees')
    ax.axis('off')  # Remove axes
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    if output_pdf is not None:
        plt.savefig(output_pdf, bbox_inches='tight')
        plt.close(fig)

    return R_pipe


if __name__ == "__main__":
    plt.close('all')

    # Example usage
    latitude = 35.6870
    longitude = -105.9378
    elevation_MSL = 2000.0
    
    R = 1.0  # Radius of the cylinder
    local_tz = pytz.timezone('America/Denver')

    # shadow_positions = []
    # sun_vectors = []
    
    # Calculate shadow positions and sun vectors for a range of times
    # start_hour_local_time = 9  # 9 AM local time
    # end_hour_local_time = 19   # 7 PM local time
    # N_points_per_hour = 4      # Number of points per hour

    # start_time = local_tz.localize(datetime(2025, 6, 21, start_hour_local_time, 0, 0))
    # end_time = local_tz.localize(datetime(2025, 6, 21, end_hour_local_time, 0, 0))
    # time_interval = timedelta(minutes=60 // N_points_per_hour)
    
    # current_time = start_time
    # while current_time <= end_time:
    #     datetime_UTC = current_time.astimezone(pytz.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
    #     azimuth, elevation = get_solar_angle(latitude, longitude, elevation_MSL, datetime_UTC)
    #     x_shadow, y_shadow, z_shadow, sun_vector = calculate_shadow_position(latitude, R, azimuth, elevation)
    #     shadow_positions.append((x_shadow, y_shadow, z_shadow))
    #     sun_vectors.append(sun_vector)
    #     current_time += time_interval
    
    # # Plot the cylinder with shadow positions and sun vectors
    # plot_cylinder_with_shadows(latitude, R, shadow_positions, sun_vectors)
    
    # Plot the sun path for the same time of day for a range of days of the year
    day_0 = 0
    day_f = 181
    local_hours = np.arange(9, 17)
    
    sun_path_by_hour_dict = {}
    for local_hour in tqdm(local_hours):
        for d in range(day_0, day_f + 1):
            autumn_equinox_dt = local_tz.localize(datetime(2024, 12, 21, local_hour, 0))
            
            this_day = autumn_equinox_dt + timedelta(days=d)
            utc_str = this_day.astimezone(pytz.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
            az, el = get_solar_angle(latitude, longitude, elevation_MSL, utc_str)
            sx, sy, sz, sv = calculate_shadow_position(latitude, R, az, el)
            
            if local_hour not in sun_path_by_hour_dict:
                sun_path_by_hour_dict[local_hour] = {'sun_path_positions': [],
                                                     'sun_path_vectors': [],
                                                     'dt_list': [],
                                                    }
            sun_path_by_hour_dict[local_hour]['sun_path_positions'].append((sx, sy, sz))
            sun_path_by_hour_dict[local_hour]['sun_path_vectors'].append(sv)
            sun_path_by_hour_dict[local_hour]['dt_list'].append(this_day)
    
    
    # Plot sundial paths on paper and save as PDF
    W = 11.0  # Width of the paper in inches
    H = 8.5   # Height of the paper in inches
    pipe_diameter = 4.026

    output_pdf = 'sundial.pdf'
    plot_sundial_on_paper(W, H, R, pipe_diameter, sun_path_by_hour_dict, latitude, longitude, local_tz, output_pdf)
    plot_sundial_on_paper(W, H, R, pipe_diameter, sun_path_by_hour_dict, latitude, longitude, local_tz, output_pdf=None)

    plot_cylinder_with_shadows_by_hour(latitude, R, sun_path_by_hour_dict)

    plt.show()
