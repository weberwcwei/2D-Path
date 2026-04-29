# 2D Path Problem

Check whether a path exists between two points on a binary obstacle map (visibility graph + Dijkstra).

## Quick peek
See [`src/assignment.ipynb`](src/assignment.ipynb) for a walkthrough of the approach and results

## Install

```bash
uv venv
uv pip install -r ../requirements.txt
```
## Download Testcases
```bash
git clone https://github.com/mcollinswisc/2D_paths.git testcases
```

## Run

```bash
# case 1
uv run python src/poly_2d_path.py --img-path ./testcases/small-ring.png --start 0,0 --end 4,3

# case 2
uv run python src/poly_2d_path.py --img-path ./testcases/polygons.png --start 2,2 --end 95,92 

# case 3
uv run python src/poly_2d_path.py --img-path ./testcases/bars.png --start 0,0 --end 8,8   
```

Options:
- `--img-path` input map (non-zero pixels = obstacle)
- `--start` start point as `x,y`
- `--end` end point as `x,y`
- `--save-img` output image path (default: `<img-stem>_result.png`)
