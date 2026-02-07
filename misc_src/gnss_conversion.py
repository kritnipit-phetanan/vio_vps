import math, csv

WGS84_A = 6378137.0
WGS84_F = 1/298.257223563
WGS84_E2 = WGS84_F*(2 - WGS84_F)

def dm_to_deg(dm_str, hemi):
    # 'ddmm.mmmm' or 'dddmm.mmmm' -> decimal degrees with sign from hemisphere
    dm = float(dm_str)
    deg = int(dm // 100)
    minutes = dm - deg*100
    out = deg + minutes/60.0
    if hemi in ('S','W'):
        out = -out
    return out

def parse_gga(sentence):
    # sentence like "$GNGGA,191116.20,4728.82000,N,05300.56771,W,4,12,0.66,32.5,M,7.8,M,1.2,0000*7A"
    if '*' in sentence:
        sentence = sentence.split('*',1)[0]  # drop checksum for parsing
    parts = sentence.split(',')
    if len(parts) < 15 or parts[0][-3:] != 'GGA':
        return None
    lat = dm_to_deg(parts[2], parts[3])
    lon = dm_to_deg(parts[4], parts[5])
    alt_msl_m = float(parts[9]) if parts[10] == 'M' and parts[9] != '' else float('nan')
    geoid_sep_m = float(parts[11]) if parts[12] == 'M' and parts[11] != '' else float('nan')
    # Ifอยากได้ ellipsoid height: h = H + N
    alt_ellip_m = alt_msl_m + geoid_sep_m if (not math.isnan(alt_msl_m) and not math.isnan(geoid_sep_m)) else float('nan')
    return lat, lon, alt_msl_m, alt_ellip_m

def geodetic_to_ecef(lat_deg, lon_deg, h_m):
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    sinp, cosp = math.sin(phi), math.cos(phi)
    sinl, cosl = math.sin(lam), math.cos(lam)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * sinp*sinp)
    x = (N + h_m)*cosp*cosl
    y = (N + h_m)*cosp*sinl
    z = (N*(1 - WGS84_E2) + h_m)*sinp
    return x,y,z

def ecef_to_enu(x,y,z, x0,y0,z0, lat0_deg, lon0_deg):
    phi0 = math.radians(lat0_deg); lam0 = math.radians(lon0_deg)
    sinp0, cosp0 = math.sin(phi0), math.cos(phi0)
    sinl0, cosl0 = math.sin(lam0), math.cos(lam0)
    dx, dy, dz = x - x0, y - y0, z - z0
    # E,N,U
    e = -sinl0*dx + cosl0*dy
    n = -sinp0*cosl0*dx - sinp0*sinl0*dy + cosp0*dz
    u =  cosp0*cosl0*dx + cosp0*sinl0*dy + sinp0*dz
    return e,n,u

def central_diff(vals, times):
    # vals: list of numbers; times: list of seconds (same length)
    n = len(vals)
    v = [float('nan')]*n
    if n == 0: return v
    if n >= 2:
        v[0] = (vals[1]-vals[0]) / (times[1]-times[0])
        v[-1] = (vals[-1]-vals[-2]) / (times[-1]-times[-2])
    for i in range(1, n-1):
        dt = (times[i+1]-times[i-1])
        v[i] = (vals[i+1]-vals[i-1]) / dt
    return v

def mph(mps): return mps*2.236936

# ---- main: read CSV, compute ENU & speeds ----
rows = []
with open('nmea.csv','r',newline='') as f:
    r = csv.DictReader(f)
    for d in r:
        sent = d.get('sentence','')
        t = float(d.get('stamp_log'))  # seconds (ROS stamp)
        p = parse_gga(sent)
        if p:
            lat, lon, alt_msl_m, alt_ellip_m = p
            # ใช้ alt_msl_m เป็น U (Up) เพื่อตรงกับ AGL/sea level pipeline
            rows.append({'t':t, 'lat':lat, 'lon':lon, 'h_msl':alt_msl_m})

if len(rows) < 2:
    raise SystemExit("ต้องมีอย่างน้อย 2 บรรทัด GGA เพื่อนับความเร็ว")

# origin = แถวแรก
lat0 = rows[0]['lat']; lon0 = rows[0]['lon']; h0 = rows[0]['h_msl']
x0,y0,z0 = geodetic_to_ecef(lat0, lon0, h0)

E,N,U,T = [],[],[],[]
for r in rows:
    x,y,z = geodetic_to_ecef(r['lat'], r['lon'], r['h_msl'])
    e,n,u = ecef_to_enu(x,y,z, x0,y0,z0, lat0, lon0)
    E.append(e); N.append(n); U.append(u); T.append(r['t'])

vE = central_diff(E,T)
vN = central_diff(N,T)
vU = central_diff(U,T)

out = []
for i,r in enumerate(rows):
    out.append({
        'stamp_log': T[i],
        'lat_dd': r['lat'],
        'lon_dd': r['lon'],
        'altitude_MSL_m': r['h_msl'],
        'xSpeed_mph': mph(vE[i]),
        'ySpeed_mph': mph(vN[i]),
        'zSpeed_mph': mph(vU[i]),
    })

# เขียนผลลัพธ์
with open('flight_log_from_gga.csv','w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)

print("wrote flight_log_from_gga.csv with xSpeed/ySpeed/zSpeed (mph)")
