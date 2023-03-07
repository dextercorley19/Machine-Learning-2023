import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
import fastf1 as ff1
from fastf1.core import Laps
from fastf1 import utils
from fastf1 import plotting
plotting.setup_mpl()
from timple.timedelta import strftimedelta

ff1.Cache.enable_cache('cache')

Bahrain_qualification = ff1.get_session(2023, 'Bahrain', 'Q')
print(Bahrain_qualification.date)

Bahrain_qualification.load();
Bahrain_qualification.results[:3]

Bahrain_race = ff1.get_session(2023, 'Bahrain', 'R')

Bahrain_race.load();
laps_r = Bahrain_race.laps
laps_r
laps_r.columns

fastest_lap = laps_r.pick_fastest()
fastest_lap['Driver']


