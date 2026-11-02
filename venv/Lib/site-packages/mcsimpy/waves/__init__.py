"""
Waves (:mod:`mcsimpy.waves`)
================================

Waves package for simulations.

Contents
--------

 - wave_loads.py: Wave load calculations
 - wave_spectra.py: Wave spectra module

Examples
--------

>>> from mcsimpy.waves import JONSWAP, WaveLoad
"""

from mcsimpy.waves.wave_loads import WaveLoad, FluidMemory
from mcsimpy.waves.wave_spectra import JONSWAP, ModifiedPiersonMoskowitz