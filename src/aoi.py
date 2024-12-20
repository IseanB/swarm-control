import time
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def polygon_contains_point(point, polygon_vertices):
	point = Point(point[0], point[1])
	polygon = Polygon(polygon_vertices)
	return polygon.contains(point)

def define_polygon(environment, num_pts=4):
    def tellme(s):
        print(s)
        plt.title(s, fontsize=16)
        plt.draw()

    ax = plt.gca()
    plt.setp(ax, autoscale_on=0)
    x, y = environment.get_size()
    ax.set_xlim([0, x])
    ax.set_ylim([0, y])

    tellme('You will define a flight area with %d points.\nClick to begin.'%num_pts)

    plt.waitforbuttonpress()

    while True:
        pts = []
        while len(pts) < num_pts:
            tellme('Select %d corners with mouse'%num_pts)
            pts = np.asarray(plt.ginput(num_pts, timeout=-1))
            if len(pts) < num_pts:
                tellme('Too few points, starting over')
                time.sleep(1)  # Wait a second

        ph = plt.fill(pts[:, 0], pts[:, 1], 'y', lw=2)

        tellme('Happy? Press any key for YES, mouse click for NO')

        if plt.waitforbuttonpress():
            break

	    # Get rid of fill
        for p in ph:
            p.remove()

    return pts

def define_flight_area(environment):
    plt.grid()
    while True:
        try:
            num_pts = int( input('Enter number of polygonal vertixes: ') )
            break
        except:
            print('\nPlease, enter an integer number.')
    flight_area_vertices = define_polygon(environment, num_pts)
    plt.clf()
    plt.grid()
    return flight_area_vertices