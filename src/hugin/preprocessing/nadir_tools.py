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


def get_nadir_status(**kw):
    angle = int(kw["OffNadirAngle"])
    if angle >= 0 and angle <= 25:
        return "Nadir"
    elif angle >= 26 and angle <= 40:
        return "OffNadir"
    elif angle >= 41 and angle <= 55:
        return "VeryOffNadir"
    else:
        raise NotImplementedError("Nadir angle not supported!")


class FilterDatasetByNadir(object):
    def __init__(self, nadir_type=None):
        self._nadir_type = nadir_type

    def __call__(self, dataset_id, match_components, dataset):
        return self._nadir_type == match_components["nadir"]
