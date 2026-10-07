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
COLUMNS = ('sim_time_s', 'x_m', 'y_m', 'yaw_rad', 'speed_mps')


def write_trace_csv(trace, path: Path) -> None:
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(trace)


def _closest_sample(trace, x: float, y: float):
    """The trace row where the robot came closest to (x, y)."""
    return min(trace, key=lambda row: math.hypot(row[1] - x, row[2] - y))


def path_chart(map_yaml: Path, route, trace, path: Path) -> None:
    """The robot's path drawn over the route's map, with its start and waypoints."""
    meta = yaml.safe_load(Path(map_yaml).read_text())
    image = np.asarray(Image.open(Path(map_yaml).parent / meta['image']).convert('L'))
    resolution = float(meta['resolution'])
    origin_x, origin_y = (float(v) for v in meta['origin'][:2])

    xs = [row[1] for row in trace]
    ys = [row[2] for row in trace]
    waypoints = list(route.waypoints)
    fig = go.Figure()
    # Row 0 of the image is the map's top; flip it so the heatmap grows upwards from origin.
    fig.add_trace(go.Heatmap(
        z=np.flipud(image), x0=origin_x + resolution / 2, dx=resolution,
        y0=origin_y + resolution / 2, dy=resolution, zmin=0, zmax=255,
        colorscale='gray', showscale=False, hoverinfo='skip', name='map'))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name='robot',
                             line=dict(color='#d62728', width=2)))
    fig.add_trace(go.Scatter(x=[route.initial_pose.x], y=[route.initial_pose.y], mode='markers',
                             name='start', marker=dict(symbol='star', size=14, color='#2ca02c')))
    fig.add_trace(go.Scatter(x=[p.x for p in waypoints], y=[p.y for p in waypoints],
                             mode='markers+text', name='waypoints',
                             text=[str(i + 1) for i in range(len(waypoints))], textposition='top center',
                             marker=dict(size=11, color='#1f77b4')))
    all_x = xs + [p.x for p in waypoints] + [route.initial_pose.x]
    all_y = ys + [p.y for p in waypoints] + [route.initial_pose.y]
    margin = 3.0
    fig.update_layout(
        title=f'{route.name}: robot path on the map', template='plotly_white',
        xaxis=dict(title='x (m)', range=[min(all_x) - margin, max(all_x) + margin]),
        yaxis=dict(title='y (m)', range=[min(all_y) - margin, max(all_y) + margin],
                   scaleanchor='x', scaleratio=1),
    )
    fig.write_html(path)


def speed_chart(trace, route, path: Path) -> None:
    """Speed over sim time, marked where the robot passed each waypoint."""
    t0 = trace[0][0]
    fig = go.Figure(go.Scatter(x=[row[0] - t0 for row in trace], y=[row[4] for row in trace],
                               mode='lines', name='speed'))
    for i, waypoint in enumerate(route.waypoints):
        passed = _closest_sample(trace, waypoint.x, waypoint.y)
        fig.add_vline(x=passed[0] - t0, line_dash='dash', line_color='#1f77b4',
                      annotation_text=f'wp {i + 1}', annotation_position='top')
    fig.update_layout(title=f'{route.name}: speed', xaxis_title='sim time (s)',
                      yaxis_title='speed (m/s)', template='plotly_white')
    fig.write_html(path)


def waypoint_chart(result, path: Path) -> None:
    """Per waypoint, from follow_route(): closest approach, lidar clearance and time."""
    labels = [f'wp {w.index + 1}' for w in result.waypoints]
    fig = go.Figure([
        go.Bar(name='closest approach (m)', x=labels, y=[w.closest_approach_m for w in result.waypoints]),
        go.Bar(name='min clearance (m)', x=labels, y=[w.min_clearance_m for w in result.waypoints]),
        go.Bar(name='reached after (s)', x=labels, y=[w.time_s for w in result.waypoints], yaxis='y2',
               opacity=0.5),
    ])
    fig.update_layout(
        title=f'{result.label or result.route}: waypoints ({result.outcome})', barmode='group',
        template='plotly_white', yaxis=dict(title='metres'),
        yaxis2=dict(title='seconds', overlaying='y', side='right', showgrid=False),
    )
    fig.write_html(path)
