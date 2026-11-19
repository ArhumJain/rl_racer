"""Batched tensor versions of the ``racer_rl.Track`` queries the racing env makes each step.

Every query is computed in float64, whatever precision the car dynamics run in: the
centerline arcs have radii up to about 30 km, and their centres are that far away,
so float32 would lose millimetres in the lateral offset.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from highway_env.road.lane import CircularLane

from racer_rl import Track

LOCATE_AHEAD = 25.0
""" How far ahead of the hint ``Track.locate`` looks for the car [m] """


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


class TrackTables:
    """The track's centerline arcs and widths as tensors, with batched queries."""

    def __init__(self, track: Track, device: torch.device | str) -> None:
        lanes = track.lanes
        if not all(isinstance(lane, CircularLane) for lane in lanes):
            raise NotImplementedError("only tracks made of circular arcs are supported")
        as_tensor = lambda values, dtype=torch.float64: torch.as_tensor(
            np.asarray(values), dtype=dtype, device=device
        )
        self.length = track.length
        self.starts = as_tensor(track.starts)
        self.lengths = as_tensor([lane.length for lane in lanes])
        self.centers = as_tensor([lane.center for lane in lanes])
        self.radii = as_tensor([lane.radius for lane in lanes])
        self.start_phases = as_tensor([lane.start_phase for lane in lanes])
        self.directions = as_tensor([lane.direction for lane in lanes])
        self.curvatures = as_tensor(track.curvatures)
        self.candidates = as_tensor(_locate_candidates(self.lengths.cpu().numpy()), torch.long)
        # np.interp(period=L) pads the grid with one point on each side, one period away
        grid, widths = track._grid, track.half_widths
        self.width_grid = as_tensor(np.concatenate([grid[-1:] - track.length, grid, grid[:1] + track.length]))
        self.width_values = as_tensor(np.concatenate([widths[-1:], widths, widths[:1]]))

    def segment_at(self, s: torch.Tensor) -> torch.Tensor:
        return torch.searchsorted(self.starts, torch.remainder(s, self.length), right=True) - 1

    def _along(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Segment index, and the angle on its circle, of the arc lengths ``s``."""
        i = self.segment_at(s)
        longitudinal = torch.remainder(s, self.length) - self.starts[i]
        return i, self.directions[i] * longitudinal / self.radii[i] + self.start_phases[i]

    def position(self, s: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
        i, phi = self._along(s)
        distance = self.radii[i] - lateral * self.directions[i]
        return self.centers[i] + distance[:, None] * torch.stack([torch.cos(phi), torch.sin(phi)], -1)

    def heading(self, s: torch.Tensor) -> torch.Tensor:
        i, phi = self._along(s)
        return phi + math.pi / 2 * self.directions[i]

    def curvature(self, s: torch.Tensor) -> torch.Tensor:
        return self.curvatures[self.segment_at(s)]

    def half_width(self, s: torch.Tensor) -> torch.Tensor:
        x = torch.remainder(s, self.length)
        j = torch.searchsorted(self.width_grid, x, right=True) - 1
        x0, x1 = self.width_grid[j], self.width_grid[j + 1]
        y0, y1 = self.width_values[j], self.width_values[j + 1]
        return (y1 - y0) / (x1 - x0) * (x - x0) + y0

    def locate(
        self, position: torch.Tensor, hint: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched ``Track.locate``: (segment, arc length s, lateral offset) of each car."""
        candidates = self.candidates[hint]
        delta = position[:, None, :] - self.centers[candidates]
        start_phase, direction = self.start_phases[candidates], self.directions[candidates]
        radius, length = self.radii[candidates], self.lengths[candidates]
        # highway-env's CircularLane.local_coordinates and its L1 CircularLane.distance
        phi = torch.atan2(delta[..., 1], delta[..., 0])
        phi = start_phase + wrap_to_pi(phi - start_phase)
        longitudinal = direction * (phi - start_phase) * radius
        lateral = direction * (radius - torch.sqrt(delta[..., 0] ** 2 + delta[..., 1] ** 2))
        distance = lateral.abs() + torch.relu(longitudinal - length) + torch.relu(-longitudinal)
        best = distance.argmin(dim=1, keepdim=True)
        segment = candidates.gather(1, best)[:, 0]
        longitudinal = longitudinal.gather(1, best)[:, 0]
        clamped = torch.minimum(torch.clamp(longitudinal, min=0.0), self.lengths[segment])
        s = torch.remainder(self.starts[segment] + clamped, self.length)
        return segment, s, lateral.gather(1, best)[:, 0]


def _locate_candidates(lengths: np.ndarray) -> np.ndarray:
    """For each hint, the segments ``Track.locate`` compares, padded by repeating the last."""
    n = len(lengths)
    rows = []
    for hint in range(n):
        candidates = [(hint - 1) % n, hint]
        ahead = 0.0
        while ahead < LOCATE_AHEAD and len(candidates) < n:
            candidates.append((candidates[-1] + 1) % n)
            ahead += lengths[candidates[-1]]
        rows.append(candidates)
    width = max(len(row) for row in rows)
    return np.array([row + row[-1:] * (width - len(row)) for row in rows])
