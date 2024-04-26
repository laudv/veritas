# from matplotsoccer
# https://github.com/TomDecroos/matplotsoccer/blob/master/matplotsoccer/fns.py

import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import numpy as np
import math

spadl_config = {
    "length": 105,
    "width": 68,
    "penalty_box_length": 16.5,
    "penalty_box_width": 40.3,
    "six_yard_box_length": 5.5,
    "six_yard_box_width": 18.3,
    "penalty_spot_distance": 11,
    "goal_width": 7.3,
    "goal_length": 2,
    "origin_x": 0,
    "origin_y": 0,
    "circle_radius": 9.15,
}

zline = 8000
zfield = -5000
zheatmap = 7000
zaction = 9000
ztext = 9500

def _plot_rectangle(x1, y1, x2, y2, ax, color):
    ax.plot([x1, x1], [y1, y2], color=color, zorder=zline)
    ax.plot([x2, x2], [y1, y2], color=color, zorder=zline)
    ax.plot([x1, x2], [y1, y1], color=color, zorder=zline)
    ax.plot([x1, x2], [y2, y2], color=color, zorder=zline)
    
def field(
    ax,
    linecolor="black",
    fieldcolor="white",
    alpha = 1.0,
    field_config=spadl_config,
):
    cfg = field_config

    # Pitch Outline & Centre Line
    x1, y1, x2, y2 = (
        cfg["origin_x"],
        cfg["origin_y"],
        cfg["origin_x"] + cfg["length"],
        cfg["origin_y"] + cfg["width"],
    )

    d = cfg["goal_length"]
    rectangle = plt.Rectangle(
        (x1 - 2 * d, y1 - 2 * d),
        cfg["length"] + 4 * d,
        cfg["width"] + 4 * d,
        fc=fieldcolor,
        alpha=alpha,
        zorder=zfield,
    )
    ax.add_patch(rectangle)
    _plot_rectangle(x1, y1, x2, y2, ax=ax, color=linecolor)
    ax.plot([(x1 + x2) / 2, (x1 + x2) / 2], [y1, y2], color=linecolor, zorder=zline)

    # Left Penalty Area
    x1 = cfg["origin_x"]
    x2 = cfg["origin_x"] + cfg["penalty_box_length"]
    m = (cfg["origin_y"] + cfg["width"]) / 2
    y1 = m - cfg["penalty_box_width"] / 2
    y2 = m + cfg["penalty_box_width"] / 2
    _plot_rectangle(x1, y1, x2, y2, ax=ax, color=linecolor)

    # Right Penalty Area
    x1 = cfg["origin_x"] + cfg["length"] - cfg["penalty_box_length"]
    x2 = cfg["origin_x"] + cfg["length"]
    m = (cfg["origin_y"] + cfg["width"]) / 2
    y1 = m - cfg["penalty_box_width"] / 2
    y2 = m + cfg["penalty_box_width"] / 2
    _plot_rectangle(x1, y1, x2, y2, ax=ax, color=linecolor)

    # Left 6-yard Box
    x1 = cfg["origin_x"]
    x2 = cfg["origin_x"] + cfg["six_yard_box_length"]
    m = (cfg["origin_y"] + cfg["width"]) / 2
    y1 = m - cfg["six_yard_box_width"] / 2
    y2 = m + cfg["six_yard_box_width"] / 2
    _plot_rectangle(x1, y1, x2, y2, ax=ax, color=linecolor)

    # Right 6-yard Box
    x1 = cfg["origin_x"] + cfg["length"] - cfg["six_yard_box_length"]
    x2 = cfg["origin_x"] + cfg["length"]
    m = (cfg["origin_y"] + cfg["width"]) / 2
    y1 = m - cfg["six_yard_box_width"] / 2
    y2 = m + cfg["six_yard_box_width"] / 2
    _plot_rectangle(x1, y1, x2, y2, ax=ax, color=linecolor)

    # Left Goal
    x1 = cfg["origin_x"] - cfg["goal_length"]
    x2 = cfg["origin_x"]
    m = (cfg["origin_y"] + cfg["width"]) / 2
    y1 = m - cfg["goal_width"] / 2
    y2 = m + cfg["goal_width"] / 2
    _plot_rectangle(x1, y1, x2, y2, ax=ax, color=linecolor)

    # Right Goal
    x1 = cfg["origin_x"] + cfg["length"]
    x2 = cfg["origin_x"] + cfg["length"] + cfg["goal_length"]
    m = (cfg["origin_y"] + cfg["width"]) / 2
    y1 = m - cfg["goal_width"] / 2
    y2 = m + cfg["goal_width"] / 2
    _plot_rectangle(x1, y1, x2, y2, ax=ax, color=linecolor)

    # Prepare Circles
    mx, my = (cfg["origin_x"] + cfg["length"]) / 2, (cfg["origin_y"] + cfg["width"]) / 2
    centreCircle = plt.Circle(
        (mx, my), cfg["circle_radius"], color=linecolor, fill=False, zorder=zline
    )
    centreSpot = plt.Circle((mx, my), 0.4, color=linecolor, zorder=zline)

    lx = cfg["origin_x"] + cfg["penalty_spot_distance"]
    leftPenSpot = plt.Circle((lx, my), 0.4, color=linecolor, zorder=zline)
    rx = cfg["origin_x"] + cfg["length"] - cfg["penalty_spot_distance"]
    rightPenSpot = plt.Circle((rx, my), 0.4, color=linecolor, zorder=zline)

    # Draw Circles
    ax.add_patch(centreCircle)
    ax.add_patch(centreSpot)
    ax.add_patch(leftPenSpot)
    ax.add_patch(rightPenSpot)

    # Prepare Arcs
    r = cfg["circle_radius"] * 2
    leftArc = Arc(
        (lx, my),
        height=r,
        width=r,
        angle=0,
        theta1=307,
        theta2=53,
        color=linecolor,
        zorder=zline,
    )
    rightArc = Arc(
        (rx, my),
        height=r,
        width=r,
        angle=0,
        theta1=127,
        theta2=233,
        color=linecolor,
        zorder=zline,
    )

    # Draw Arcs
    ax.add_patch(leftArc)
    ax.add_patch(rightArc)