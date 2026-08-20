import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io
import math
import folium
from streamlit_folium import st_folium

# --- 1. CONFIGURATIE & UI ---
st.set_page_config(page_title="Franse Huizen Zoeker", layout="wide")
st.title("🏡 Franse Huizen Geolocation Tool (PoC)")
st.markdown("Upload een foto, geef het middelpunt van de zoekcirkel op en de tool zoekt naar matchende kenmerken (zoals zwembaden) via Franse IGN-satellietbeelden.")

# --- 2. HULPFUNCTIES VOOR COÖRDINATEN EN TEGELS ---
def deg2num(lat_deg, lon_deg, zoom):
    """Zet GPS-coördinaten om naar de juiste tegel (X, Y) op de kaart."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    """Zet een tegel (X, Y) terug om naar GPS-coördinaten (Noordwest hoek)."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

# --- 3. SATELLIET EN BEELDHERKENNING ---
def fetch_ign_satellite_tile(xtile, ytile, zoom):
    """Haalt de luchtfoto op van de Franse overheid (IGN)."""
    url = f"https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=ORTHOIMAGERY.ORTHOPHOTOS&STYLE=normal&FORMAT=image/jpeg&TILEMATRIXSET=PM&TILEMATRIX={zoom}&TILEROW={ytile}&TILECOL={xtile}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
    except Exception as e:
        st.error(f"Fout bij ophalen tegel: {e}")
    return None

def find_swimming_pools(pil_image):
    """Zoekt naar zwembad-blauw in een afbeelding en retourneert de pixel-coördinaten."""
    img = np.array(pil_image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Kleurbereik voor zwembadblauw
    lower_blue = np.array([80, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    pools = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 15 < area < 500: # Filter op logische grootte (voorkomt ruis)
            x, y, w, h = cv2.boundingRect(cnt)
            pools.append({"x": x + w/2, "y": y + h/2}) # Middelpunt van zwembad
    return pools

# --- 4. SIDEBAR INPUTS ---
with st.sidebar:
    st.header("📍 Zoekparameters")
    uploaded_image = st.file_uploader("1. Upload Makelaarsfoto (Optioneel voor nu)", type=["jpg", "jpeg", "png"])
    
    st.subheader("2. Locatie van de cirkel")
    lat_input = st.number_input("Latitude (bijv. 44.837)", value=44.837789, format="%.6f")
    lon_input = st.number_input("Longitude (bijv. -0.579)", value=-0.579180, format="%.6f")
    
    grid_size = st.slider("Zoekgebied grootte (aantal tegels)", min_value=1, max_value=5, value=3, step=2, help="Een raster van 3x3 tegels om het middelpunt.")
    
    start_search = st.button("Start Zoekopdracht 🚀", type="primary")

# --- 5. HOOFDLOGICA EN WEERGAVE ---
if start_search:
    st.info("Satellietbeelden ophalen en scannen op kenmerken (zwembaden)...")
    
    # Bepaal de centrale tegel op Zoomniveau 17 (geschikt voor huizen)
    ZOOM = 17
    center_xtile, center_ytile = deg2num(lat_input, lon_input, ZOOM)
    
    found_locations = []
    offset = grid_size // 2
    
    # Progress bar
    progress_bar = st.progress(0)
    total_tiles = grid_size * grid_size
    current_tile = 0
    
    # Scan een grid rondom het middelpunt
    for dx in range(-offset, offset + 1):
        for dy in range(-offset, offset + 1):
            current_xtile = center_xtile + dx
            current_ytile = center_ytile + dy
            
            tile_img = fetch_ign_satellite_tile(current_xtile, current_ytile, ZOOM)
            
            if tile_img:
                pools_in_tile = find_swimming_pools(tile_img)
                
                # Bereken benaderde GPS coördinaten voor de gevonden zwembaden
                if pools_in_tile:
                    nw_lat, nw_lon = num2deg(current_xtile, current_ytile, ZOOM)
                    se_lat, se_lon = num2deg(current_xtile + 1, current_ytile + 1, ZOOM)
                    
                    # 256x256 is de standaard pixel-grootte van een IGN tile
                    for pool in pools_in_tile:
                        # Interpolatie van pixels naar GPS coördinaten
                        pool_lon = nw_lon + (pool["x"] / 256.0) * (se_lon - nw_lon)
                        pool_lat = nw_lat + (pool["y"] / 256.0) * (se_lat - nw_lat)
                        found_locations.append((pool_lat, pool_lon))
            
            current_tile += 1
            progress_bar.progress(current_tile / total_tiles)
            
    st.success(f"Scan voltooid! Er zijn {len(found_locations)} potentiële matches gevonden in het gebied.")
    
    # --- 6. KAART GENEREREN ---
    m = folium.Map(location=[lat_input, lon_input], zoom_start=15)
    
    # Teken de zoekcirkel (grove indicatie van het gescande gebied)
    folium.Circle(
        location=[lat_input, lon_input],
        radius=grid_size * 150, # ruwe schatting in meters
        color="blue",
        fill=True,
        fill_color="#3186cc"
    ).add_to(m)
    
    # Plaats markers op de gevonden locaties
    for loc in found_locations:
        folium.Marker(
            location=[loc[0], loc[1]],
            popup="🔍 Potentiële Match",
            icon=folium.Icon(color="green", icon="home")
        ).add_to(m)
        
    st_folium(m, width=1000, height=600)

elif not start_search:
    st.write("👈 Gebruik het menu aan de linkerkant om de coördinaten in te voeren en de scan te starten.")
