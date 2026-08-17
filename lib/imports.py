import sys, os          
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import trange
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from collections import defaultdict
import scipy.stats
import matplotlib.gridspec as gridspec
from scipy.stats import chi2_contingency
import random
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, ScalarFormatter)
import matplotlib.patches as mpatches
font = {'family' : 'serif', 'size' : 14 }
mpl.rc('font', **font)
mpl.rcParams['mathtext.fontset'] = 'cm' # Set the math font to Computer Modern
mpl.rcParams['legend.fontsize'] = 16
# Set default figure and axes face colors to white
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
