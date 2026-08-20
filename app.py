import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io
import math
import base64
import json
import re
import folium
from streamlit_folium import st_folium
from openai import OpenAI

# --- 1. CONFIGURATIE & UI ---
st.set_page_config(page_title="Franse Huizen Zoeker AI", layout="wide")
st.title("🏡 Franse Huizen Geolocation Tool (met GPT-4o Vision)")
st.markdown("Upload buitenfoto's van een makelaars-advertentie en geef de locatie op via URL, plaatsnaam of coördinaten.")

# OpenAI Client setup
api_key = st.secrets.get("OPENAI_API_KEY", None)
if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None
    st.warning("⚠️ Geen `OPENAI_API_KEY` gevonden in Streamlit Secrets. AI-fotoanalyse staat uit totdat de sleutel is ingesteld.")

# --- 2. HULPFUNCTIES VOOR LOCATIE EN URL ---
def extract_coords_from_url(url: str):
    """Haalt automatisch coördinaten uit een Bien'ici of Google Maps URL."""
    match_bienici = re.search(r'camera=\d+_([0-9.-]+)_([0-9.-]+)', url)
    if match_bienici:
        lon = float(match_bienici.group(1))
        lat = float(match_bienici.group(2))
        return lat, lon
    
    match_gmaps = re.search(r'@([0-9.-]+),([0-9.-]+)', url)
    if match_gmaps:
        return float(match_gmaps.group(1)), float(match_gmaps.group(2))
        
    return None

def geocode_french_town(town_name: str):
    """Zet een Franse plaatsnaam/postcode om naar GPS-coördinaten via de overheids-API."""
    url = f"https://api-adresse.data.gouv.fr/search/?q={town_name}&limit=1"
    try:
        res = requests.get(url, timeout=5).json()
        if res.get('features'):
            coords = res['features'][0]['geometry']['coordinates']
            return coords[1], coords[0]
    except Exception:
        pass
    return None

# --- 3. AI FOTO-ANALYSE FUNCTIE ---
def analyze_photo_with_gpt4o(image_bytes):
    """Gebruikt GPT-4o Vision om de foto te analyseren op satelliet-herkenbare kenmerken."""
    if not client:
        return None

    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    prompt = """
    Analyseer deze buitenfoto van een woning in Frankrijk voor satelliet-geolocatie matching.
    Geef ALLEEN een geldig JSON object terug met de volgende velden (in het Nederlands):
    {
      "dak_kleur": "bijv. rood/oranje dakpannen, donkere leisteen, grijs",
      "dak_vorm": "bijv. rechthoekig, L-vormig, complex, schuin",
      "heeft_zwembad": true/false,
      "bijgebouwen": "bijv. vrijstaande schuur aanwezig, garage vast aan huis, geen",
      "omgeving_en_groen": "bijv. omringd door bomen/bos, open veld, tuin met gazon",
      "oprit_en_terrein": "bijv. onverharde oprit aan de zijkant, binnenplaats/cour",
      "unieke_kenmerken": ["lijst", "van", "opvallende", "kenmerken"]
    }
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=500
        )
        result_text = response.choices[0].message.content
        return json.loads(result_text)
    except Exception as e:
        st.error(f"Fout bij AI analyse: {e}")
        return None

# --- 4. IGN SATELLIET & MATH HULPFUNCTIES ---
def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def fetch_ign_satellite_tile(xtile, ytile, zoom):
    url = f"https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=ORTHOIMAGERY.ORTHOPHOTOS&STYLE=normal&FORMAT=image/jpeg&TILEMATRIXSET=PM&TILEMATRIX={zoom}&TILEROW={ytile}&TILECOL={xtile}"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            return Image.open(io.BytesIO(res.content))
    except Exception:
        pass
    return None

def scan_tile_features(pil_image, ai_analysis):
    """Scant de satellietfoto op basis van AI-kenmerken."""
    img = np.array(pil_image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    matches = []

    # Zoek naar zwembaden als AI een zwembad ziet
    if ai_analysis and ai_analysis.get("heeft_zwembad"):
        lower_blue = np.array([80, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if 15 < cv2.contourArea(cnt) < 500:
                x, y, w, h = cv2.boundingRect(cnt)
                matches.append({"x": x + w/2, "y": y + h/2, "type": "Zwembad Match"})

    # Zoek naar rode/oranje daken
    dak_kleur = str(ai_analysis.get("dak_kleur", "")).lower() if ai_analysis else ""
    if "rood" in dak_kleur or "oranje" in dak_kleur or "dakpan" in dak_kleur:
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = mask1 | mask2
        contours, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < 2000:
                x, y, w, h = cv2.boundingRect(cnt)
                matches.append({"x": x + w/2, "y": y + h/2, "type": "Gebouw/Dak Match"})

    return matches

# --- 5. SIDEBAR INPUTS (CORRECT INGESPRONGEN) ---
with st.sidebar:
    st.header("1. Upload Makelaarsfoto's")
    uploaded_file = st.file_uploader("Kies een buitenfoto van het huis", type=["jpg", "jpeg", "png"])
    
    st.header("2. Locatie bepalen")
    location_method = st.radio("Kies invoermethode:", ["Link/URL plakken", "Plaatsnaam / Postcode", "Handmatig Lat/Lon"])
    
    lat_input, lon_input = 44.891237, 1.832689  # Standaard coördinaten uit jouw Bien'ici voorbeeld
    
    if location_method == "Link/URL plakken":
        listing_url = st.text_input("Plak de advertentie URL (bijv. van Bien'ici):")
        if listing_url:
            extracted = extract_coords_from_url(listing_url)
            if extracted:
                lat_input, lon_input = extracted
                st.success(f"📍 Coördinaten gevonden: {lat_input:.5f}, {lon_input:.5f}")
            else:
                st.warning("Kon geen automatische coördinaten in deze URL vinden.")
                
    elif location_method == "Plaatsnaam / Postcode":
        town_input = st.text_input("Plaatsnaam of postcode:", value="Aynac 46120")
        if town_input:
            geo_coords = geocode_french_town(town_input)
            if geo_coords:
                lat_input, lon_input = geo_coords
                st.success(f"📍 Centrum van {town_input}: {lat_input:.5f}, {lon_input:.5f}")
            else:
                st.error("Plaatsnaam niet gevonden.")

    elif location_method == "Handmatig Lat/Lon":
        lat_input = st.number_input("Latitude", value=44.891237, format="%.6f")
        lon_input = st.number_input("Longitude", value=1.832689, format="%.6f")

    grid_size = st.slider("Zoekbereik (grid grootte)", min_value=1, max_value=7, value=3, step=2)
    start_search = st.button("Start AI Analyse & Zoekopdracht 🚀", type="primary")

# --- 6. HOOFDLOGICA ---
ai_data = None

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(uploaded_file, caption="Geüploade Woningfoto", use_container_width=True)
    
    with col2:
        if client:
            with st.spinner("🤖 GPT-4o analyseert de foto op satellietkenmerken..."):
                file_bytes = uploaded_file.getvalue()
                ai_data = analyze_photo_with_gpt4o(file_bytes)
            
            if ai_data:
                st.subheader("📊 AI Geolocatie Profiel")
                st.json(ai_data)
        else:
            st.info("Voeg de OpenAI API Key toe om AI-fotoanalyse in te schakelen.")

if start_search:
    st.divider()
    st.subheader("🔍 Satellietbeelden scannen via Franse Overheid (IGN)...")
    
    ZOOM = 17
    center_x, center_y = deg2num(lat_input, lon_input, ZOOM)
    found_hits = []
    offset = grid_size // 2
    
    progress = st.progress(0)
    total_tiles = grid_size * grid_size
    step = 0
    
    for dx in range(-offset, offset + 1):
        for dy in range(-offset, offset + 1):
            tx = center_x + dx
            ty = center_y + dy
            tile_img = fetch_ign_satellite_tile(tx, ty, ZOOM)
            
            if tile_img:
                hits = scan_tile_features(tile_img, ai_data)
                if hits:
                    nw_lat, nw_lon = num2deg(tx, ty, ZOOM)
                    se_lat, se_lon = num2deg(tx + 1, ty + 1, ZOOM)
                    for hit in hits:
                        hit_lon = nw_lon + (hit["x"] / 256.0) * (se_lon - nw_lon)
                        hit_lat = nw_lat + (hit["y"] / 256.0) * (se_lat - nw_lat)
                        found_hits.append({"lat": hit_lat, "lon": hit_lon, "type": hit["type"]})
            
            step += 1
            progress.progress(step / total_tiles)
            
    st.success(f"Scan voltooid! Er zijn {len(found_hits)} potentiële locaties gedetecteerd.")
    
    # Kaart weergave
    m = folium.Map(location=[lat_input, lon_input], zoom_start=15)
    folium.Circle(location=[lat_input, lon_input], radius=grid_size * 180, color="blue", fill=True, fill_color="#3186cc", fill_opacity=0.1).add_to(m)
    
    for hit in found_hits:
        folium.Marker(
            location=[hit["lat"], hit["lon"]],
            popup=f"📍 Match: {hit['type']}",
            icon=folium.Icon(color="red" if "Zwembad" in hit["type"] else "green", icon="home")
        ).add_to(m)
        
    st_folium(m, width=1000, height=600)
