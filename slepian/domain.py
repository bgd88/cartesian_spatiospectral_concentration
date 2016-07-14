import numpy as np
import numpy.ma as ma

class Subdomain(object):

    def __init__(self, extent):
        """
        Base class initializer, taking the extents of
        the total domain, in the format (width, height).
        The total domain then exists in the area (0, width)x(0,height).
        This should be called by derived classes.
        """
        self._extent = extent

    def in_subdomain(self, x, y):
        """
        Returns True or False if the point x,y
        is in the subdomain.
        """
        raise NotImplemented

    def _extent_function(self):
        """
        Returns a tuple of the width, height
        of the domain in which the subdomain lives.
        """
        raise NotImplemented

    def _compute_area(self):
        """
        Compute the area of the subdomain.
        The base class implementation computes
        it by  approximate numerical quadrature,
        but base classes may override this with
        analytical formulae if they are known.
        """
        width,height = self.extent
        nx,ny = (1000,1000)
        x = np.linspace(0., width, nx)
        y = np.linspace(0., height, ny)
        dx = width/nx
        dy = height/ny
        xgrid,ygrid = np.meshgrid(x,y)
        
        subdomain = ma.masked_where( ~self.in_subdomain(xgrid,ygrid), np.ones_like(xgrid))
        area = np.sum(subdomain)*dx*dy
        return area

    @property
    def area(self):
        return self._compute_area()

    @property
    def extent(self):
        return self._extent


class Disc(Subdomain):
    def __init__( self, extent, center, radius):
        """
        Initialize a 2D disc subdomain, with extents=(width,height)
        of the total domain, center=(x0,y0) of the center point of
        the disc, and radius is the radius of the disc.
        """
        self._center = center
        self._radius = radius
        Subdomain.__init__(self, extent)

    def in_subdomain(self, x, y):
        r = self._radius
        x0,y0 = self._center[0], self._center[1]
        return (x-x0)*(x-x0) + (y-y0)*(y-y0) <= r*r

    def _compute_area(self):
        return self._radius*self._radius*np.pi

class Rectangle(Subdomain):
    def __init__( self, extent, lower_left, upper_right):
        """
        Initialize a 2D rectangle subdomain.
        """
        assert (lower_left[0] < upper_right[0])
        assert (lower_left[1] < upper_right[1])
        self.lower_left = lower_left
        self.upper_right = upper_right
        Subdomain.__init__(self, extent)

    def in_subdomain(self, x, y):
        return np.logical_and( np.logical_and( x > self.lower_left[0], x < self.upper_right[0]),
               np.logical_and(y > self.lower_left[1], y < self.upper_right[1]) )

    def _compute_area(self):
        return (self.upper_right[0] - self.lower_left[0])*\
               (self.upper_right[1] - self.lower_left[1])
