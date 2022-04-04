'''
Analysis on the follow target module behavior through ulog file.
'''
import numpy as np
import pandas as pd
from pyulog import ULog
import plotly.graph_objects as go
from typing import Tuple
import argparse

###############
# Aux functions
def get_follow_target_status_pose_angle_data(msg_data_dict):
    # Get the data length for this log
    data_len = len(msg_data_dict['timestamp'])
    print('Follow Target Status : Data Length :', data_len)

    pos_0 = msg_data_dict['pos_est_filtered[0]'].reshape([data_len, 1])
    pos_1 = msg_data_dict['pos_est_filtered[1]'].reshape([data_len, 1])
    pos_2 = msg_data_dict['pos_est_filtered[2]'].reshape([data_len, 1])

    pos_concatenated = np.concatenate((pos_0, pos_1, pos_2), axis = 1)

    vel_0 = msg_data_dict['vel_est_filtered[0]'].reshape([data_len, 1])
    vel_1 = msg_data_dict['vel_est_filtered[1]'].reshape([data_len, 1])
    vel_2 = msg_data_dict['vel_est_filtered[2]'].reshape([data_len, 1])

    vel_concatenated = np.concatenate((vel_0, vel_1, vel_2), axis = 1)

    orbit_angles = msg_data_dict['orbit_angle_setpoint']
    try:
        tracked_target_orientations = msg_data_dict['tracked_target_course']
    except:
        tracked_target_orientations = msg_data_dict['tracked_target_orientation']

    return (pos_concatenated, vel_concatenated, tracked_target_orientations, orbit_angles)



def get_xyz_vxyz_data(msg_data_dict):
    # Get the data length for this log
    data_len = len(msg_data_dict['timestamp'])
    print('Data Length :', data_len)

    pos_0 = msg_data_dict['x'].reshape([data_len, 1])
    pos_1 = msg_data_dict['y'].reshape([data_len, 1])
    pos_2 = msg_data_dict['z'].reshape([data_len, 1])

    pos_concatenated = np.concatenate((pos_0, pos_1, pos_2), axis = 1)

    vel_0 = msg_data_dict['vx'].reshape([data_len, 1])
    vel_1 = msg_data_dict['vy'].reshape([data_len, 1])
    vel_2 = msg_data_dict['vz'].reshape([data_len, 1])

    vel_concatenated = np.concatenate((vel_0, vel_1, vel_2), axis = 1)

    return (pos_concatenated, vel_concatenated)



# https://docs.scipy.org/doc/scipy/tutorial/interpolate.html
def continuous_to_discrete_interpolate(cont_timevectors, cont_values, discrete_timevectors):
    # If the continous values is a vector (on each timevector idx), break it down into each one-dimensional
    # Arrays and concatenate them to return as a final discrete interpolation.
    if(len(cont_values.shape) > 1): # If it's not (N, ) shape
        cont_values_2nd_dim = cont_values.shape[1] # Get the 2nd dim size
        return_val = continuous_to_discrete_interpolate(cont_timevectors, cont_values[:,0], discrete_timevectors)
        return_val = np.reshape(return_val, [return_val.shape[0], 1]) # Set 2nd dimension into '1' to allow concatenation

        for dim in range(1, cont_values_2nd_dim):
            one_dim_return = continuous_to_discrete_interpolate(cont_timevectors, cont_values[:,dim], discrete_timevectors)
            one_dim_return = np.reshape(one_dim_return, [one_dim_return.shape[0], 1]) # Set 2nd dimension into '1' to allow concatenation
            return_val = np.concatenate((return_val, one_dim_return), axis=0)

        return return_val

    x = cont_timevectors
    y = cont_values
    y_new = np.interp(discrete_timevectors, x, y)
    return y_new

###########
# Functions
def analyze_and_visualize_log(log_fname):
    #################
    # Analyze the log

    ulog = ULog(log_fname)
    data_list = ulog.data_list

    MESSAGES_LEN = 100
    follow_target_timestamps = np.ndarray((MESSAGES_LEN, 1))
    follow_target_pos_filtered = np.ndarray((MESSAGES_LEN, 3))
    follow_target_vel_filtered = np.ndarray((MESSAGES_LEN, 3))
    follow_target_target_orientations = np.ndarray((MESSAGES_LEN, 1))
    follow_target_current_orbit_angles = np.ndarray((MESSAGES_LEN, 1))

    vehicle_local_pose_timestamps = np.ndarray((MESSAGES_LEN, 1))
    vehicle_local_pos = np.ndarray((MESSAGES_LEN, 3))
    vehicle_local_vel = np.ndarray((MESSAGES_LEN, 3))

    # Velocity setpoint visualization (Commanded by Follow Target Flight Task)
    vehicle_local_pose_setpoint_timestamps = np.ndarray((MESSAGES_LEN, 1))
    vehicle_local_pos_setpoint = np.ndarray((MESSAGES_LEN, 3))
    vehicle_local_vel_setpoint = np.ndarray((MESSAGES_LEN, 3))

    # Trajectory setpiont visualization (Commanded by MPC Position controller)
    trajectory_setpoint_timestamps = np.ndarray((MESSAGES_LEN, 1))
    trajectory_setpoint_pos = np.ndarray((MESSAGES_LEN, 3))
    trajectory_setpoint_vel = np.ndarray((MESSAGES_LEN, 3))

    # Go through the all the different messages and pick the one we want
    for data in data_list:
        msg_name = data.name
        msg_data = data.data
        if msg_name == 'follow_target_status':
            follow_target_timestamps = msg_data['timestamp']
            follow_target_pos_filtered, follow_target_vel_filtered, follow_target_target_orientations, follow_target_current_orbit_angles = get_follow_target_status_pose_angle_data(msg_data)
        elif msg_name == 'vehicle_local_position':
            vehicle_local_pose_timestamps = msg_data['timestamp']
            vehicle_local_pos, vehicle_local_vel = get_xyz_vxyz_data(msg_data)
        elif msg_name == 'vehicle_local_position_setpoint':
            vehicle_local_pose_setpoint_timestamps = msg_data['timestamp']
            vehicle_local_pos_setpoint, vehicle_local_vel_setpoint = get_xyz_vxyz_data(msg_data)
        elif msg_name == 'trajectory_setpoint': # From Follow target Flight Task
            trajectory_setpoint_timestamps = msg_data['timestamp']
            trajectory_setpoint_pos, trajectory_setpoint_vel = get_xyz_vxyz_data(msg_data)


    # End of data processing

    ####################
    # Process timestamps
    timestamp_start = max(follow_target_timestamps[0], vehicle_local_pose_timestamps[0])
    timestamp_end = min(follow_target_timestamps[-1], vehicle_local_pose_timestamps[-1])

    # Shift timestamp start to the point where the target position is not NaN
    timestamp_idx_start_new = 0
    while True:
        if np.isnan(np.sum(follow_target_pos_filtered[timestamp_idx_start_new])): # Check if any element is NaN
            timestamp_idx_start_new += 1
            #print(follow_target_pos_filtered[timestamp_idx_start_new])
        else:
            break
    # Pick the 'latest' timestamp, since it guarantees the validity of having both vehicle position & follow target position data,
    # as well as the follow target position data not being NaN.
    timestamp_start = max(timestamp_start, follow_target_timestamps[timestamp_idx_start_new])
    print('timestamp_start :', timestamp_start, 'end :', timestamp_end)

    # Create timevectors starting from 0 (roughly), in unit of seconds.
    # Can't subtract with timestamp_start, since it will cause underflow, in case the subtraction results in a negative number.
    # timestamp_min = min(follow_target_timestamps[0], vehicle_local_pose_timestamps[0]) # This will be our 'base' timestamp, to process all the timestamps.
    # timestamp_max = max(follow_target_timestamps[-1], vehicle_local_pose_timestamps[-1])

    follow_target_timevectors = np.divide(follow_target_timestamps, 1E6)
    vehicle_local_pose_timevectors = np.divide(vehicle_local_pose_timestamps, 1E6)

    # Timestamps separated via 0.01 seconds
    time_step = 0.01
    time_vector = np.arange((timestamp_start)/1E6, (timestamp_end)/1E6, time_step)
    len_time = len(time_vector)

    ###############
    # GRAPH Variables
    target_position = np.zeros([2, len_time])
    target_velocity  = np.zeros([2, len_time])

    position_setpoint_calculated = np.zeros([2, len_time])
    ft_vel_setpoint = np.zeros([2, len_time])

    raw_follow_position = np.zeros([2, len_time])
    commanded_follow_position = np.zeros([2, len_time])

    follow_angle = np.zeros(len_time)
    target_orientation = np.zeros(len_time)
    orbit_angle = np.zeros(len_time)
    orbit_tangential_velocity = np.zeros([2, len_time])

    # Follow distance [m]
    follow_distance = 8.

    #####################
    # Discretize the data
    target_orientation = continuous_to_discrete_interpolate(follow_target_timevectors, follow_target_target_orientations, time_vector)
    orbit_angle = continuous_to_discrete_interpolate(follow_target_timevectors, follow_target_current_orbit_angles, time_vector)
    target_position = continuous_to_discrete_interpolate(follow_target_timevectors, follow_target_pos_filtered[:,0:2], time_vector).reshape([2, len_time])
    target_velocity = continuous_to_discrete_interpolate(follow_target_timevectors, follow_target_vel_filtered[:,0:2], time_vector).reshape([2, len_time])

    ft_pos_setpoint = continuous_to_discrete_interpolate(trajectory_setpoint_timestamps / 1E6, trajectory_setpoint_pos[:,0:2], time_vector).reshape([2, len_time])
    ft_vel_setpoint = continuous_to_discrete_interpolate(trajectory_setpoint_timestamps / 1E6, trajectory_setpoint_vel[:,0:2], time_vector).reshape([2, len_time])

    local_pos_setpoint = continuous_to_discrete_interpolate(vehicle_local_pose_setpoint_timestamps / 1E6, vehicle_local_pos_setpoint[:,0:2], time_vector).reshape([2, len_time])
    local_vel_setpoint = continuous_to_discrete_interpolate(vehicle_local_pose_setpoint_timestamps / 1E6, vehicle_local_vel_setpoint[:,0:2], time_vector).reshape([2, len_time])

    # orbit angle projection
    position_setpoint_calculated[:, :] = target_position[:, :] + \
        follow_distance * np.array([np.cos(orbit_angle[:]), np.sin(orbit_angle[:])])

    real_position = continuous_to_discrete_interpolate(vehicle_local_pose_timevectors, vehicle_local_pos[:,0:2], time_vector).reshape([2, len_time])

    ###############
    # GRAPH Drawing

    # sample data at time interval
    slider_time_interval = 0.5  # s

    # the index interval for the chosen slider time interval
    k_interval = int(slider_time_interval / time_step)

    num_slider_frames = int(np.floor(len_time / k_interval)) + 1

    # this sets the duration each frame takes when animating with the 'play' button [ms]
    # (slider_time_interval is 'real time')
    step_duration = 0.5 * slider_time_interval * 1e3

    # this sets the duration each frame takes when manually moving the slider [ms]
    slider_step_duration = 20

    # PLOTTING OPTIONS

    # plot tail histories on positions
    plot_tails = False  # XXX: these don't work as expected fully ATM
    len_tail_s = 10.  # [s]
    len_tail_k = int(len_tail_s / time_step)

    # make the figure (it is just a dict!)
    fig_dict = {
        'data': [],
        'layout': {},
        'frames': []
    }

    # fill in most of the layout
    fig_dict['layout']['xaxis'] = {'title': 'X Position [m]'}
    fig_dict['layout']['yaxis'] = {'title': 'Y Position [m]'}
    fig_dict['layout']['hovermode'] = 'closest'
    fig_dict['layout']['updatemenus'] = [
    {
        'buttons': [
            # play button
            {
                    'args': [
                        None,
                        {
                            'frame': {'duration': step_duration, 'redraw': False},
                            'fromcurrent': True,
                            'transition': {'duration': step_duration, 'easing': 'linear'}
                        }
                    ],
                    'label': 'Play',
                    'method': 'animate'
                },
                # pause button
                {
                    'args': [
                        [None],
                        {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }
                    ],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            # 'direction': 'left',
            # 'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            # 'x': 0.1,
            # 'xanchor': 'right',
            # 'y': 0,
            # 'yanchor': 'top'
    }
    ]

    # empty sliders dict
    sliders_dict = {
        'active': 0,  # the initially active frame
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'visible': True
        },
        # 'transition': {'duration': step_duration, 'easing': 'linear'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': []
    }   

    xaxis_max = np.max([np.max(target_position[1, :]), np.max(raw_follow_position[1, :]), np.max(position_setpoint_calculated[1, :])])
    xaxis_min = np.min([np.min(target_position[1, :]), np.min(raw_follow_position[1, :]), np.min(position_setpoint_calculated[1, :])])
    yaxis_max = np.max([np.max(target_position[0, :]), np.max(raw_follow_position[0, :]), np.max(position_setpoint_calculated[0, :])])
    yaxis_min = np.min([np.min(target_position[0, :]), np.min(raw_follow_position[0, :]), np.min(position_setpoint_calculated[0, :])])

    print('xaxis min max :', xaxis_min, xaxis_max, 'yaxis min max :', yaxis_min, yaxis_max)

    # make frames
    for k in range(0, len_time, k_interval):

        # initialize the frame
        frame = {'data': [], 'name': f'{time_vector[k]:3.1f}'}

        # tail indices
        k_start = max(k - len_tail_k, 0)
        k_end = min(k + 1, len_time)

        # populate the frame data list with traces

        # marker for target position
        frame['data'].append(
            go.Scatter(
                x = [target_position[1, k]],
                y = [target_position[0, k]],
                marker = dict({
                    'symbol': 'circle',
                    'size': 15,
                    'color': 'rgb(180, 180, 180)'  # gray
                }),
                name = 'Target position filtered'
            )
        )

        # orbit perimeter around target position
        orbit_perimeter_y = [target_position[0, k] + follow_distance * np.cos(angle) for angle in np.linspace(-np.pi, np.pi, 201)]
        orbit_perimeter_x = [target_position[1, k] + follow_distance * np.sin(angle) for angle in np.linspace(-np.pi, np.pi, 201)]
        frame['data'].append(
            go.Scatter(
                x = orbit_perimeter_x,
                y = orbit_perimeter_y,
                line = dict({
                    'color': 'rgb(180, 180, 180)'  # gray
                }),
                name = 'Orbit perimeter'
            )
        )

        # target orientation
        frame['data'].append(
            go.Scatter(
                x = np.array([0., np.sin(target_orientation[k]) * follow_distance]) + target_position[1, k],
                y = np.array([0., np.cos(target_orientation[k]) * follow_distance]) + target_position[0, k],
                name = 'Target orientation',
                line = dict({
                    'color': 'rgb(100, 100, 100)',
                    'dash': 'dashdot'
                })
            )
        )

        # TARGET SPEED
        # ------------
        # Visualize the velocity vector coming out of target position
        frame['data'].append(
            go.Scatter(
                x = np.array([0., target_velocity[1, k]]) + target_position[1, k],
                y = np.array([0., target_velocity[0, k]]) + target_position[0, k],
                name = 'Target Speed',
                line = dict({
                    'color': 'rgb(0, 255, 0)',
                    'dash': 'dashdot'
                })
            )
        )

        # FOLLOW TARGET COMMANDS
        # - - - - - - - -

        # marker for position setpoint
        frame['data'].append(
            go.Scatter(
                x = [ft_pos_setpoint[1, k]],
                y = [ft_pos_setpoint[0, k]],
                marker = dict({
                    'symbol': 'circle-open',
                    'size': 15,
                    'color': 'rgb(0,0,255)',
                }),
                name = 'Position setpoint by Follow Target'
            )
        )

         # Velocity setpoint - coming out of 'follow target position setpoint'
        frame['data'].append(
            go.Scatter(
                x = np.array([0., ft_vel_setpoint[1, k]]) + ft_pos_setpoint[1, k],
                y = np.array([0., ft_vel_setpoint[0, k]]) + ft_pos_setpoint[0, k],
                name = 'Velocity setpoint by Follow Target',
                line = dict({
                    'color': 'rgb(0, 0, 255)',
                    'dash': 'dashdot'
                })
            )
        )

        # LOCAL POSE SETPOINT
        # ------------
        # Visualize the trajectory velocity setpiont generated by MPC Position Controller, coming out of drone's actual position

        frame['data'].append(
            go.Scatter(
                x = np.array([0., local_vel_setpoint[1, k]]) + real_position[1, k],
                y = np.array([0., local_vel_setpoint[0, k]]) + real_position[0, k],
                name = 'Velocity setpoint by Position Controller',
                line = dict({
                    'color': 'rgb(255, 0, 0)',
                    'dash': 'dashdot'
                })
            )
        )

        # Visualize Trajectory position setpoint
        frame['data'].append(
            go.Scatter(
                x = [local_pos_setpoint[1, k]],
                y = [local_pos_setpoint[0, k]],
                marker = dict({
                    'symbol': 'circle-open',
                    'size': 20,
                    'color': 'rgb(255,0,0)'
                }),
                name = 'Position setpoint by Position Controller'
            )
        )

        # DRONE ACTUAL POSITION
        # - - - - - - - - - -
        # marker for raw follow position
        frame['data'].append(
            go.Scatter(
                x = [real_position[1, k]],
                y = [real_position[0, k]],
                marker = dict({
                    'symbol': 'circle',
                    'size': 20,
                    'color': 'rgb(0,255,0)'
                }),
                name = 'Drone Actual Position'
            )
        )

        # append the frame
        fig_dict['frames'].append(frame)

        # set up the slider step corresponding to this frame
        slider_step = {
            'args': [
                [frame['name']],
                {
                    'frame': {'duration': slider_step_duration, 'redraw': False},
                    'mode': 'afterall',
                    'transition': {'duration': slider_step_duration}
                }
            ],
            'label': frame['name'],
            'method': 'animate'
        }

        # append the slider step
        sliders_dict['steps'].append(slider_step)

    # set the sliders
    fig_dict['layout']['sliders'] = [sliders_dict]

    # use the first frame for initial data view
    fig_dict['data'] = fig_dict['frames'][0]['data']

    fig = go.Figure(fig_dict)

    # dimensions + title
    fig.update_layout(
        title_text = 'Follow-target log data Visualization\nLogFile : {}'.format(log_fname)
    )

    # Fix x axis & y axis scale to same value
    # https://plotly.com/python/axes/#fixed-ratio-axes
    fig.update_yaxes (
        scaleanchor = "x",
        scaleratio = 1,
    )

    # fix the xy range
    fig['layout']['xaxis']['range'] = [xaxis_min, xaxis_max]
    fig['layout']['yaxis']['range'] = [yaxis_min, yaxis_max]

    fig.show()


# Main Function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='PX4 Follow Target log analyzer')
    parser.add_argument(
        '--log_path',
        dest='log_path',
        required=True,
        help='Path to where the ULog to be analyzed is located')

    args = parser.parse_args()

    analyze_and_visualize_log(args.log_path)
