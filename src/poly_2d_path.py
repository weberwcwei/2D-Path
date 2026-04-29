import numpy as np
import imageio.v3 as imageio
import matplotlib.pyplot as plt
import heapq
import math
import argparse
from pathlib import Path

def find_corners(universe: np.ndarray) -> list[tuple[int, int]]:
    '''Find convex corners of the obstacles.

    example:
    0 0 1     . . x
    0 1 0  => . P .
    0 0 0     . . .

    . = free, P = free corner, x = obstacle
    '''
    w, h = universe.shape
    obs = universe.astype(bool)
    free = ~obs

    P = np.pad(obs, 1, mode="constant", constant_values=False)
    # neighbor at offset (di, dj) of original (x, y) is P[1+di+x, 1+dj+y]
    NW = P[0:w,     0:h]      # (-1, -1)
    N  = P[0:w,     1:h+1]    # (-1,  0)
    NE = P[0:w,     2:h+2]    # (-1, +1)
    W  = P[1:w+1,   0:h]      # ( 0, -1)
    E  = P[1:w+1,   2:h+2]    # ( 0, +1)
    SW = P[2:w+2,   0:h]      # (+1, -1)
    S  = P[2:w+2,   1:h+1]    # (+1,  0)
    SE = P[2:w+2,   2:h+2]    # (+1, +1)
    corner = free & ((NW & ~N & ~W) | (NE & ~N & ~E) | (SW & ~S & ~W) | (SE & ~S & ~E))
    xs, ys = np.where(corner)
    return list(zip(xs.tolist(), ys.tolist()))


def is_reachable(universe, x0, y0, x1, y1) -> bool:
    '''Check if start and end are in the same connected component of free space.'''
    W, H = universe.shape
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1    # step direction in x
    sy = 1 if y0 < y1 else -1    # step direction in y
    err = dx - dy
    x, y = x0, y0

    while True:
        # out of bounds or obstacle
        if not (0 <= x < W and 0 <= y < H) or universe[x, y] == 1:   
            return False
        if x == x1 and y == y1:    # reached destination
            return True

        err2 = err * 2  # Bresenham's line algorithm
        step_x = err2 > -dy
        step_y = err2 <  dx
        # corner crossing check
        if step_x and step_y:
            if universe[x + sx, y] == 1 or universe[x, y + sy] == 1:
                return False
        if step_x:
            err -= dy
            x += sx
        if step_y:
            err += dx
            y += sy


def path_exists(
        universe: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        visualize: bool = True,
        save_path: str | None = None) -> bool:
    '''Check if path exists from start to end in the map
    1. Find all corners of obstacle
    2. Build visibility graph using corners
    3. Dijkstra find shortest path (edges weighted by Euclidean distance)
    '''
    W, H = universe.shape
    start_x, start_y = start
    end_x, end_y = end

    def show(path):    # draw map + start/end, plus path if found
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(universe, cmap="gray", origin="upper")
        if path is not None:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(ys, xs, c="cyan", lw=2, marker="o", ms=4, zorder=2, label=f"path ({len(path)} nodes)")
            ax.set_title("path found")
        else:
            ax.set_title("no path")
        ax.scatter(start_y, start_x, c="lime", s=120, marker="o", edgecolors="black", label=f"start {start}", zorder=3)
        ax.scatter(end_y,   end_x,   c="red",  s=120, marker="X", edgecolors="black", label=f"end {end}",     zorder=3)
        ax.set_xlabel("y (col)")
        ax.set_ylabel("x (row)")
        ax.legend(loc="upper right")
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

     # start and end are the same point
    if (start_x, start_y) == (end_x, end_y):   
        if visualize: show([start])
        return True
    for x, y in [(start_x, start_y), (end_x, end_y)]:
        # out of bounds or obstacle
        if not (0 <= x < W and 0 <= y < H) or universe[x, y] == 1:    
            if visualize: show(None)
            return False

    nodes = [(start_x, start_y), (end_x, end_y)]
    nodes += find_corners(universe)
    
    seen, uniq = set(), []
    for nd in nodes:
        if nd not in seen:
            seen.add(nd)
            uniq.append(nd)
    
    nodes = uniq
    n = len(nodes)

    # Dijkstra
    START, END = 0, 1
    dist    = [math.inf] * n
    parent  = [-1]       * n    # for path reconstruction when visualize=True
    settled = [False]    * n
    dist[START] = 0.0
    pq = [(0.0, START)]          # (distance, node index)

    while pq:
        d, u = heapq.heappop(pq)
        if settled[u]:
            continue
        settled[u] = True

        if u == END:
            if visualize:
                path, cur = [], END
                while cur != START:
                    path.append(nodes[cur])
                    cur = parent[cur]
                path.append(nodes[START])
                path.reverse()
                show(path)
            return True

        for v in range(n):
            if settled[v]:
                continue
            if is_reachable(universe, *nodes[u], *nodes[v]):
                nd = d + math.dist(nodes[u], nodes[v])
                if nd < dist[v]:
                    dist[v]   = nd
                    parent[v] = u
                    heapq.heappush(pq, (nd, v))

    if visualize: show(None)
    return False


def parse_xy(s: str) -> tuple[int, int]:
    '''Parse "x,y" or "x y" into (int, int).'''
    parts = s.replace(",", " ").split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected 'x,y', got {s!r}")
    return int(parts[0]), int(parts[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-path", required=True, help="Path to obstacle map image.")
    parser.add_argument("--start", required=True, type=parse_xy, help='Start point as "x,y".')
    parser.add_argument("--end",   required=True, type=parse_xy, help='End point as "x,y".')
    parser.add_argument("--save-img", default=None, help="Output image path. Default: <img-path stem>_result.png")
    args = parser.parse_args()

    img_path = Path(args.img_path)
    arr = (imageio.imread(img_path) > 0).astype(np.uint8)
    out = Path(args.save_img) if args.save_img else img_path.with_name(img_path.stem + "_result.png")

    result = path_exists(arr, args.start, args.end, visualize=True, save_path=str(out))
    print(f"{args.start} -> {args.end}: {'reachable' if result else 'no path'}  saved: {out}")