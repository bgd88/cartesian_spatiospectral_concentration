import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

def create_basemap(map_range, resolution='l'):
    """Create basemap instance from map range."""
    center = [(map_range[1]+map_range[0])/2.0, (map_range[3]+map_range[2])/2.0]
    basemap_map_instance = \
        Basemap(projection='lcc',
                lat_0=center[0], lon_0=center[1],
                llcrnrlat=map_range[0], urcrnrlat=map_range[1],
                llcrnrlon=map_range[2], urcrnrlon=map_range[3],
                resolution=resolution)
    return basemap_map_instance


def read_lat_lon_domain_perimeter(perimeterFile):
    """Get slab perimeter from file."""
    columns = ['lon', 'lat']
    perimeter = pd.read_table(perimeterFile, sep=' ',
                              skipinitialspace=True,
                              names=columns, header=None)
    map_range = np.array([min(perimeter['lat']), max(perimeter['lat']), 
                          min(perimeter['lon']), max(perimeter['lon'])])
    bmap = create_basemap(map_range)
    perimeter['x'], perimeter['y'] = bmap(perimeter['lon'].values, 
                                          perimeter['lat'].values) 
    return perimeter, bmap

perimeterFile = "cas_slab1.0.clip"
perimeter, bmap  = read_lat_lon_domain_perimeter(perimeterFile)
nx = 50
ny = 50
x_pts = np.linspace(min(perimeter['x']), max(perimeter['x']), nx)
y_pts = np.linspace(min(perimeter['y']), max(perimeter['y']), ny)
x_grid, y_grid = np.meshgrid(x_pts, y_pts)
grid_pts = np.dstack([x_grid, y_grid]).reshape(nx*ny,2)
perimeter_verts = [p for p in zip(perimeter['x'].values, 
                                  perimeter['y'].values)]
path_codes = np.insert(4*np.ones(len(perimeter_verts)-1),0,1)
perimeter_path = Path(perimeter_verts, path_codes)
patch = patches.PathPatch(perimeter_path, facecolor='none', lw=2)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(grid_pts[:,0], grid_pts[:,1])
ax.add_patch(patch)
plt.show()

out = perimeter_path.contains_points(grid_pts).reshape((nx,ny))
print(out)
print(np.sum(out))
x_grid_masked = np.ma.masked_where(~out, x_grid)
y_grid_masked = np.ma.masked_where(~out, y_grid)
z_grid_masked = np.ma.masked_where(~out, np.ones_like(x_grid))
plt.pcolormesh(x_grid_masked, y_grid_masked, z_grid_masked)
plt.colorbar()
plt.show()
