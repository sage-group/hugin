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

import sys
import logging
from contextlib import contextmanager

from urllib.parse import urlparse


import os
import shutil

__scheme_handlers = {

}

log = logging.getLogger(__name__)


class IOUtilsType(type):
    def __getattr__(self, item):
        def __wrapper(path, *args, **kw):
            with discover_io_handler(path) as hndl:
                return getattr(hndl, item)(path, *args, **kw)

        return __wrapper

class IOUtils(metaclass=IOUtilsType):
    pass


class LocalIOUtils(object):

    @classmethod
    def open_file(cls, filename, mode="r", encoding="utf8"):
        kws = {}
        if 'b' not in mode:
            kws['encoding'] = encoding
        return open(filename, mode, **kws)

    @classmethod
    def file_exists(cls, filename):
        return os.path.exists(filename)

    @classmethod
    def delete_file(cls, filename):
        os.remove(filename)

    @classmethod
    def read_file_to_string(cls, filename, binary_mode=False):
        mode = "r" if not binary_mode else "rb"
        kws = {}
        if binary_mode:
            kws['encoding'] = "utf8"
        return cls.open_file(filename, mode, **kws).read()

    @classmethod
    def write_string_to_file(cls, filename, file_content):
        cls.open_file(filename, "wb", encoding="utf8").write(file_content)

    @classmethod
    def create_dir(cls, dirname):
        os.mkdir(dirname)

    @classmethod
    def recursive_create_dir(cls, dirname):
        os.makedirs(dirname)

    @classmethod
    def copy(cls, oldpath, newpath, overwrite=False):
        shutil.copy(oldpath, newpath)

    @classmethod
    def rename(cls, oldname, newname, overwrite=False):
        os.rename(oldname, newname)

    @classmethod
    def delete_recursively(cls, dirname):
        shutil.rmtree(dirname)

    @classmethod
    def is_directory(cls, dirname):
        return os.path.isdir(dirname)


__scheme_handlers[''] = LocalIOUtils
__scheme_handlers['file'] = LocalIOUtils

try:
    from tensorflow.python.lib.io import file_io


    class TFIOUtils(object):

        @classmethod
        def open_file(cls, filename, mode="r", encoding="utf8"):
            return file_io.FileIO(filename, mode)

        @classmethod
        def file_exists(cls, filename):
            return file_io.file_exists(filename)

        @classmethod
        def delete_file(cls, filename):
            return file_io.delete_file(filename)

        @classmethod
        def read_file_to_string(cls, filename, binary_mode=False):
            return file_io.read_file_to_string(filename, binary_mode=binary_mode)

        @classmethod
        def write_string_to_file(cls, filename, file_content):
            return file_io.write_string_to_file(filename, file_content)

        @classmethod
        def create_dir(cls, dirname):
            return file_io.create_dir(dirname)

        @classmethod
        def recursive_create_dir(cls, dirname):
            return file_io.recursive_create_dir(dirname)

        @classmethod
        def copy(cls, oldpath, newpath, overwrite=False):
            file_io.copy(oldpath, newpath, overwrite)

        @classmethod
        def rename(cls, oldname, newname, overwrite=False):
            file_io.rename(oldname, newname, overwrite)

        @classmethod
        def delete_recursively(cls, dirname):
            file_io.delete_recursively(dirname)

        @classmethod
        def is_directory(cls, dirname):
            return file_io.is_directory(dirname)


    __scheme_handlers['gs'] = TFIOUtils
    __scheme_handlers['s3'] = TFIOUtils

except ImportError:
    pass


@contextmanager
def discover_io_handler(path):
    url_components = urlparse(path)
    if not url_components.scheme:
        yield LocalIOUtils
    elif url_components.scheme not in __scheme_handlers:
        raise NotImplementedError("Unsupported scheme: %s", url_components.scheme)
    else:
        yield __scheme_handlers.get(url_components.scheme)
