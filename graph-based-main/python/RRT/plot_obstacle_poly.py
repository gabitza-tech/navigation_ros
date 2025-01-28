import numpy as np

def plot_obstacle_poly(ax, poly, color):
  shape_count = len(poly)

  for i in range(shape_count):
    crnt_poly = poly[i]
    poly_size = crnt_poly.shape

    # Check if the dimension is 2
    if poly_size[1] != 2:
      return # Error, incorrect polygon definition
    
    # Check if the line segment returns to the start
    if not np.array_equal(crnt_poly[-1, :], crnt_poly[0, :]):
      crnt_poly = np.vstack([crnt_poly, crnt_poly[0, :]])
      poly_size = crnt_poly.shape

    # Plot the polygon
    if color is None:
      ax.plot(crnt_poly[:, 0], crnt_poly[:, 1])
    else:
      ax.plot(crnt_poly[:, 0], crnt_poly[:, 1], color=color) 

    ax.set_aspect('equal', 'box')
  return