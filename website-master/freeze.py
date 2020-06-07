# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:58:43 2020

@author: nenas
"""

from flask_frozen import Freezer
from myapp import app

freezer = Freezer(app)
if __name__ == '__main__':
    freezer.freeze()