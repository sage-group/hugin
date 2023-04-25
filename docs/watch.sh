#!/bin/bash

fswatch --exclude='build.*' -o .. | xargs -n1 make html
