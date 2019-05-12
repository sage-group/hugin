# -*- coding: utf-8 -*-
__license__ = \
    """Copyright 2019 West University of Timisoara
    
       Licensed under the Apache License, Version 2.0 (the "License");
       you may not use this file except in compliance with the License.
       You may obtain a copy of the License at
    
           http://www.apache.org/licenses/LICENSE-2.0
    
       Unless required by applicable law or agreed to in writing, software
       distributed under the License is distributed on an "AS IS" BASIS,
       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
       See the License for the specific language governing permissions and
       limitations under the License.
    """

from logging import getLogger

import cv2
import numpy as np
from shapely.geometry import LineString, Point, MultiLineString
from skimage.morphology import medial_axis

log = getLogger(__name__)


def vectorize_array(input_array, transform, tolerance=0.4):
    skel, _ = medial_axis(input_array, return_distance=True)
    array = np.uint8(skel)
    im2, contours, hierarchy = cv2.findContours(array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    lines_obj = []
    hashes = {}
    for contour in contours:
        points = []
        for point_info in contour:
            point_coord = (point_info[0][0], point_info[0][1])
            # point_coord = transform * point_coord
            point_coord = (float(point_coord[0]), float(point_coord[1]))
            point = Point(point_coord)
            points.append(Point(point_coord))

        if len(points) > 1:
            line_obj = LineString(points)
            line_obj = line_obj.simplify(tolerance, preserve_topology=False)
            coords = list(line_obj.coords)
            new_points = [Point(coords.pop(0))]
            prev_point = new_points[0]
            for coord in coords:
                prev_coords = prev_point.coords[0]
                edge_in = prev_coords + coord
                edge_out = coord + prev_coords
                if edge_in in hashes or edge_out in hashes:
                    continue
                point = Point(coord)
                if point.distance(prev_point) < 0.001:
                    continue
                hashes[edge_in] = True
                hashes[edge_out] = True
                prev_point = point
                new_points.append(point)
            if len(new_points) > 1:
                new_line_obj = LineString(new_points)
            lines_obj.append(new_line_obj)
    return MultiLineString(lines_obj)


def to_text(vect, scene_id, transform, output_fd):
    if not vect.geoms:
        output_fd.write("%s,LINESTRING EMPTY\n" % scene_id)
        return
    for geom in vect.geoms:
        line_str = '%s,"%s"' % (scene_id, geom.wkt)
        output_fd.write("%s\n" % line_str)
