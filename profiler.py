import pstats
from pstats import SortKey

p = pstats.Stats('restats')
p.strip_dirs().sort_stats(SortKey.TIME).print_stats()