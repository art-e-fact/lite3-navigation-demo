"""What a route run leaves behind besides pass/fail: the robot's odom trace as CSV
and three plotly charts (path on the map, speed over time, per-waypoint figures)."""

import csv
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import yaml
from PIL import Image

# trace rows are (sim time s, x m, y m, yaw rad, speed m/s), from SimControlHarness.trace

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


def write_trace_csv(trace, path: Path) -> None:
    """Speed over sim time. Two columns only: the artefacts dashboard charts a CSV
    with exactly two columns; the x/y path is in path_chart's HTML instead."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('sim_time_s', 'speed_mps'))
        writer.writerows((row[0], row[4]) for row in trace)


def _closest_sample(trace, x: float, y: float):
    """The trace row where the robot came closest to (x, y)."""
    return min(trace, key=lambda row: math.hypot(row[1] - x, row[2] - y))


def path_chart(map_yaml: Path, route, trace, path: Path, plans=()) -> None:
    """The robot's path drawn over the route's map, with its start and waypoints, and
    the paths Nav2 planned (one per waypoint, from SimControlHarness.plans) dashed."""
    meta = yaml.safe_load(Path(map_yaml).read_text())
    image = np.asarray(Image.open(Path(map_yaml).parent / meta['image']).convert('L'))
    resolution = float(meta['resolution'])
    origin_x, origin_y = (float(coordinate) for coordinate in meta['origin'][:2])

    xs = [row[1] for row in trace]
    ys = [row[2] for row in trace]
    waypoints = list(route.waypoints)
    fig = go.Figure()
    # Row 0 of the image is the map's top; flip it so the heatmap grows upwards from origin.
    fig.add_trace(go.Heatmap(
        z=np.flipud(image), x0=origin_x + resolution / 2, dx=resolution,
        y0=origin_y + resolution / 2, dy=resolution, zmin=0, zmax=GRAY_MAX,
        colorscale='gray', showscale=False, hoverinfo='skip', name='map'))
    for i, plan in enumerate(plans):
        fig.add_trace(go.Scatter(x=[point[0] for point in plan], y=[point[1] for point in plan], mode='lines',
                                 name='planned', legendgroup='planned', showlegend=i == 0,
                                 line=dict(color=BLUE, width=LINE_WIDTH, dash='dash')))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name='robot',
                             line=dict(color=RED, width=LINE_WIDTH)))
    fig.add_trace(go.Scatter(x=[route.initial_pose.x], y=[route.initial_pose.y], mode='markers',
                             name='start', marker=dict(symbol='star', size=START_MARKER_SIZE, color=GREEN)))
    fig.add_trace(go.Scatter(x=[waypoint.x for waypoint in waypoints], y=[waypoint.y for waypoint in waypoints],
                             mode='markers+text', name='waypoints',
                             text=[str(i + 1) for i in range(len(waypoints))], textposition='top center',
                             marker=dict(size=WAYPOINT_MARKER_SIZE, color=BLUE)))
    all_x = xs + [waypoint.x for waypoint in waypoints] + [route.initial_pose.x]
    all_y = ys + [waypoint.y for waypoint in waypoints] + [route.initial_pose.y]
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
    start_time = trace[0][0]
    fig = go.Figure(go.Scatter(x=[row[0] - start_time for row in trace], y=[row[4] for row in trace],
                               mode='lines', name='speed'))
    for i, waypoint in enumerate(route.waypoints):
        passed = _closest_sample(trace, waypoint.x, waypoint.y)
        fig.add_vline(x=passed[0] - start_time, line_dash='dash', line_color=BLUE,
                      annotation_text=f'wp {i + 1}', annotation_position='top')
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
