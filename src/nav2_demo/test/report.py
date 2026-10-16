"""What a route run leaves behind besides pass/fail: the robot's odom trace as CSV
and three plotly charts (path on the map, speed over time, per-waypoint figures)."""

import csv
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import yaml
from PIL import Image

# A trace is a list of sim_control_harness.TraceSample (sim_time_s, x_m, y_m, yaw_rad, speed_mps).

# Legend across the top of the plot: the dashboard clips a legend down the right-hand side.
# entrywidth fixes each entry's width so entries cannot overlap when the viewer
# measures text before its fonts have loaded.
LEGEND_TOP = dict(orientation='h', yanchor='bottom', y=1.0, xanchor='right', x=1, entrywidth=130)

# Plotly's default blue, red and green, so the charts match the usual palette.
BLUE = '#1f77b4'
RED = '#d62728'
GREEN = '#2ca02c'
LINE_WIDTH = 2
START_MARKER_SIZE = 14  # the star is larger than the waypoint dots so it stands out
WAYPOINT_MARKER_SIZE = 11
GRAY_MAX = 255  # an 8-bit grayscale map image: 0 black (occupied) to 255 white (free)
PATH_MARGIN_M = 3.0  # empty space kept around the path and waypoints in the path chart
WALKING_MPS = 0.05  # slower than this counts as standing
MIN_SEGMENT_LENGTH_SQ = 1e-12  # two waypoints on the same spot: avoid dividing by zero
MIN_SAMPLE_INTERVAL_S = 1e-6  # two odom samples with the same stamp: avoid dividing by zero


def trace_metrics(trace, route, plans=()) -> dict:
    """Numbers for metrics.json from the odom trace and Nav2's plans, on top of
    RouteResult.metrics(). Unrounded; the test rounds when it writes."""
    path = np.array([(sample.x_m, sample.y_m) for sample in trace])
    times = np.array([sample.sim_time_s for sample in trace])
    yaws = np.unwrap([sample.yaw_rad for sample in trace])
    speeds = np.array([sample.speed_mps for sample in trace])
    walking = speeds > WALKING_MPS
    line = np.array([(pose.x, pose.y) for pose in [route.initial_pose, *route.waypoints]])
    deviation = _distance_to_line(path, line)
    walked = _polyline_length(path)
    planned = sum(_polyline_length(np.array(plan)) for plan in plans)
    yaw_rates = np.abs(np.diff(yaws) / np.maximum(np.diff(times), MIN_SAMPLE_INTERVAL_S))
    return dict(
        distance_walked_m=walked,
        mean_walking_speed_mps=speeds[walking].mean() if walking.any() else 0.0,
        max_speed_mps=speeds.max(),
        stationary_fraction=1.0 - walking.mean(),
        max_route_deviation_m=deviation.max(),
        mean_route_deviation_m=deviation.mean(),
        max_yaw_rate_rads=yaw_rates.max() if len(yaw_rates) else 0.0,
        replans=max(len(plans) - len(route.waypoints), 0),
        # walked over planned: 1 is on the plan, more is wandering; 0 when no plan was seen
        path_efficiency=walked / planned if planned else 0.0,
    )


def _polyline_length(points) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _distance_to_line(points, line):
    """Each point's distance to the nearest segment of the polyline."""
    starts, segments = line[:-1], np.diff(line, axis=0)
    from_starts = points[:, None] - starts  # each point from each segment's start
    lengths_sq = np.maximum((segments * segments).sum(-1), MIN_SEGMENT_LENGTH_SQ)
    # how far along each segment its nearest point is, from 0 (its start) to 1 (its end)
    fractions = np.clip((from_starts * segments).sum(-1) / lengths_sq, 0, 1)
    return np.linalg.norm(from_starts - fractions[..., None] * segments, axis=-1).min(axis=1)


def write_trace_csv(trace, path: Path) -> None:
    """Speed over sim time. Two columns only: the artefacts dashboard charts a CSV
    with exactly two columns; the x/y path is in path_chart's HTML instead."""
    with open(path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(('sim_time_s', 'speed_mps'))
        writer.writerows((sample.sim_time_s, sample.speed_mps) for sample in trace)


def _closest_sample(trace, x: float, y: float):
    """The trace sample where the robot came closest to (x, y)."""
    return min(trace, key=lambda sample: math.hypot(sample.x_m - x, sample.y_m - y))


def path_chart(map_yaml: Path, route, trace, path: Path, plans=()) -> None:
    """The robot's path drawn over the route's map, with its start and waypoints, and
    the paths Nav2 planned (one per waypoint, from SimControlHarness.plans) dashed."""
    meta = yaml.safe_load(Path(map_yaml).read_text())
    image = np.asarray(Image.open(Path(map_yaml).parent / meta['image']).convert('L'))
    resolution = float(meta['resolution'])
    origin_x, origin_y = (float(coordinate) for coordinate in meta['origin'][:2])

    xs = [sample.x_m for sample in trace]
    ys = [sample.y_m for sample in trace]
    waypoints = list(route.waypoints)
    waypoint_xs = [waypoint.x for waypoint in waypoints]
    waypoint_ys = [waypoint.y for waypoint in waypoints]
    fig = go.Figure()
    # Row 0 of the image is the map's top; flip it so the heatmap grows upwards from origin.
    fig.add_trace(go.Heatmap(
        z=np.flipud(image), x0=origin_x + resolution / 2, dx=resolution,
        y0=origin_y + resolution / 2, dy=resolution, zmin=0, zmax=GRAY_MAX,
        colorscale='gray', showscale=False, hoverinfo='skip', name='map'))
    for plan_index, plan in enumerate(plans):
        fig.add_trace(go.Scatter(x=[point[0] for point in plan], y=[point[1] for point in plan], mode='lines',
                                 name='planned', legendgroup='planned', showlegend=plan_index == 0,
                                 line=dict(color=BLUE, width=LINE_WIDTH, dash='dash')))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name='robot',
                             line=dict(color=RED, width=LINE_WIDTH)))
    fig.add_trace(go.Scatter(x=[route.initial_pose.x], y=[route.initial_pose.y], mode='markers',
                             name='start', marker=dict(symbol='star', size=START_MARKER_SIZE, color=GREEN)))
    fig.add_trace(go.Scatter(x=waypoint_xs, y=waypoint_ys,
                             mode='markers+text', name='waypoints',
                             text=[str(number) for number in range(1, len(waypoints) + 1)], textposition='top center',
                             marker=dict(size=WAYPOINT_MARKER_SIZE, color=BLUE)))
    all_x = xs + waypoint_xs + [route.initial_pose.x]
    all_y = ys + waypoint_ys + [route.initial_pose.y]
    fig.update_layout(
        title=f'{route.name}: robot path on the map', template='plotly_white',
        xaxis=dict(title='x (m)', range=[min(all_x) - PATH_MARGIN_M, max(all_x) + PATH_MARGIN_M]),
        yaxis=dict(title='y (m)', range=[min(all_y) - PATH_MARGIN_M, max(all_y) + PATH_MARGIN_M],
                   scaleanchor='x', scaleratio=1),
        legend=LEGEND_TOP,
    )
    fig.write_html(path)


def speed_chart(trace, route, path: Path) -> None:
    """Speed over sim time, marked where the robot passed each waypoint."""
    start_time = trace[0].sim_time_s
    fig = go.Figure(go.Scatter(x=[sample.sim_time_s - start_time for sample in trace],
                               y=[sample.speed_mps for sample in trace], mode='lines', name='speed'))
    for number, waypoint in enumerate(route.waypoints, start=1):
        passed = _closest_sample(trace, waypoint.x, waypoint.y)
        fig.add_vline(x=passed.sim_time_s - start_time, line_dash='dash', line_color=BLUE,
                      annotation_text=f'wp {number}', annotation_position='top')
    fig.update_layout(title=f'{route.name}: speed', xaxis_title='sim time (s)',
                      yaxis_title='speed (m/s)', template='plotly_white', legend=LEGEND_TOP)
    fig.write_html(path)


def waypoint_chart(result, path: Path) -> None:
    """Per waypoint, from follow_route(): how close the robot got and its lidar clearance.
    When each was reached is on the speed chart, so no second axis of seconds here."""
    labels = [f'wp {waypoint.index + 1}' for waypoint in result.waypoints]
    fig = go.Figure([
        go.Bar(name='closest approach', x=labels, y=[waypoint.closest_approach_m for waypoint in result.waypoints]),
        go.Bar(name='min clearance', x=labels, y=[waypoint.min_clearance_m for waypoint in result.waypoints]),
    ])
    fig.update_layout(
        title=f'{result.label or result.route}: waypoints ({result.outcome})', barmode='group',
        template='plotly_white', yaxis_title='metres',
        legend=LEGEND_TOP,
    )
    fig.write_html(path)
