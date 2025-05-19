import os
import time
import serial
import numpy as np
import plotly.express as px
from misc.helpful_geo_functions import get_grid_centroids_and_values

coords, values = get_grid_centroids_and_values(os.path.join('.', 'misc', 'radiation_data', 'Total_gamma_dose_rate.shp'), anchor_point = np.array([13.3977,52.4539]), granularity=150, grid_width=20000, grid_length=20000, time_offset_hours=1)

def calculate_checksum(sentence: str) -> int:
    chk = 0
    for c in sentence[1:]:
        chk ^= ord(c)
    return chk

def format_gpgga_message(lat: float, lon: float, alt: float = 0.0) -> str:
    # fixed timestamp
    ts = "120000.000"
    lat_deg = int(abs(lat))
    lat_min = (abs(lat) - lat_deg) * 60
    lat_dir = "N" if lat >= 0 else "S"
    lon_deg = int(abs(lon))
    lon_min = (abs(lon) - lon_deg) * 60
    lon_dir = "E" if lon >= 0 else "W"
    alt_str = f"{alt:.1f}"
    body = (f"$GPGGA,{ts},{lat_deg:02d}{lat_min:09.6f},{lat_dir},"
            f"{lon_deg:03d}{lon_min:09.6f},{lon_dir},1,09,2.1,"
            f"{alt_str},M,,M,,0000")
    return f"{body}*{calculate_checksum(body):02X}"

def read_odl(ser_data: serial.Serial) -> float:
    """Block until we get 6 bytes, parse mantissa/exponent, return odl."""
    while True:
        data = ser_data.read(6)
        if len(data) == 6:
            mlo, mhi = data[2], data[3]
            mant = mhi << 8 | mlo
            exp = data[4] - 256 if data[4] > 127 else data[4]
            return mant * (2 ** (exp - 15))

base = r"c:\Users\johan\programming\Simulation"
os.chdir(base)
# load grid: shape (37,37,2) where [:,:,0]=lon, [:,:,1]=lat
grid = coords
n1, n2, _ = grid.shape
odl_map = np.zeros((n1, n2), dtype=float)

# open serial ports
ser_data = serial.Serial("COM14", 4800, timeout=1)  # ODL
ser_gps  = serial.Serial("COM9",  4800, timeout=1)  # GPS emulator

for i in range(n1):
    for j in range(n2):
        lon, lat = float(grid[i,j,0]), float(grid[i,j,1])
        msg = format_gpgga_message(lat, lon, alt=90)
        ser_gps.write((msg + "\r\n").encode())
        ser_gps.flush()
        #time.sleep(0.05)
        odl = read_odl(ser_data)
        odl_map[i, j] = odl

# flatten for Plotly
lons = grid[:,:,0].ravel()
lats = grid[:,:,1].ravel()
vals = odl_map.ravel()
#save odl_map
np.save(os.path.join(base, 'odl_map_90m_above_ground_from_berlin_radiation_scenario_constant_2025_2026_23h_after_min.npy'), odl_map)

fig = px.scatter_mapbox(
    lat=lats, lon=lons, color=vals,
    color_continuous_scale="Turbo",
    mapbox_style="open-street-map",
    title="Measured ODL over Grid"
)
fig.show()
