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
st.title("🏡 Franse Huizen Geolocation Tool (Meerdere Foto's & AI)")
st.markdown("Upload meerdere buitenfoto's, kies je zoekgebied op de interactieve kaart en laat de AI scannen.")

# Initialize Session State
if "search_lat" not in st.session_state: st.session_state.search_lat = 44.891237
if "search_lon" not in st.session_state: st.session_state.search_lon = 1.832689
if "map_center" not in st.session_state: st.session_state.map_center = [44.891237, 1.832689]
if "last_town" not in st.session_state: st.session_state.last_town = ""
if "ai_data" not in st.session_state: st.session_state.ai_data = None
if "found_hits" not in st.session_state: st.session_state.found_hits = None

# OpenAI Client setup
api_key = st.secrets.get("OPENAI_API_KEY", None)
if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None
    st.warning("⚠️ Geen `OPENAI_API_KEY` gevonden in Streamlit Secrets. AI-fotoanalyse staat uit.")

# --- 2. HULPFUNCTIES VOOR LOCATIE EN URL ---
def extract_coords_from_url(url: str):
    match_bienici = re.search(r'camera=\d+_([0-9.-]+)_([0-9.-]+)', url)
    if match_bienici: return float(match_bienici.group(2)), float(match_bienici.group(1))
    match_gmaps = re.search(r'@([0-9.-]+),([0-9.-]+)', url)
    if match_gmaps: return float(match_gmaps.group(1)), float(match_gmaps.group(2))
    return None

def geocode_french_town(town_name: str):
    url = f"https://api-adresse.data.gouv.fr/search/?q={town_name}&limit=1"
    try:
        res = requests.get(url, timeout=5).json()
        if res.get('features'):
            coords = res['features'][0]['geometry']['coordinates']
            return coords[1], coords[0]
    except Exception: pass
    return None

# --- 3. AI FOTO-ANALYSE FUNCTIE ---
def analyze_photos_with_gpt4o(image_bytes_list):
    if not client or not image_bytes_list: return None
    
    prompt = """
    Jij bent een OSINT (Open Source Intelligence) expert, gespecialiseerd in het geoloceren van gebouwen op satellietbeelden.
    Analyseer deze foto's en zoek naar harde, permanente kenmerken die zichtbaar zijn vanuit de ruimte.
    
    Geef ALLEEN een JSON object terug. Baseer je antwoord strikt op wat je ziet.
    {
      "dak_kenmerken": {
          "kleur": "bijv. donkerrode dakpannen, leisteen, plat dak",
          "vorm": "bijv. rechthoekig, L-vormig, complexe aanbouw",
          "opvallende_elementen": "bijv. opvallende dakkapellen, grote schoorsteen (optioneel)"
      },
      "heeft_permanent_zwembad": true/false, // Zet op FALSE als het een opblaas- of opzetzwembad is!
      "omgeving_en_infrastructuur": {
          "verharding": "Zijn er zichtbare opritten, betonplaten of terrassen?",
          "vegetatie_patronen": "Ligt het huis direct tegen een dichte bosrand aan, of midden in een open veld?",
          "schaduw_en_helling": "Is het terrein overduidelijk geaccidenteerd/hellend?"
      },
      "osint_zoekadvies": "Geef in 2 korte zinnen gericht advies waar de satelliet-zoeker op moet letten bij het handmatig scannen van de kaart."
    }
    """
    
    messages_content = [{"type": "text", "text": prompt}]
    for img_bytes in image_bytes_list:
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        messages_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
        
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": messages_content}],
            response_format={"type": "json_object"},
            max_tokens=500
        )
        return json.loads(response.choices[0].message.content)
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
    return (math.degrees(lat_rad), lon_deg)

def fetch_ign_satellite_tile(xtile, ytile, zoom):
    url = f"https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=ORTHOIMAGERY.ORTHOPHOTOS&STYLE=normal&FORMAT=image/jpeg&TILEMATRIXSET=PM&TILEMATRIX={zoom}&TILEROW={ytile}&TILECOL={xtile}"
    try:
        res = requests.get(url, timeout=5)
        if res.status_code == 200: return Image.open(io.BytesIO(res.content))
    except: pass
    return None

def scan_tile_features(pil_image, ai_analysis):
    img = np.array(pil_image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    matches = []

    if ai_analysis and ai_analysis.get("heeft_permanent_zwembad"):
        mask = cv2.inRange(hsv, np.array([80, 50, 50]), np.array([130, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if 15 < cv2.contourArea(cnt) < 500:
                x, y, w, h = cv2.boundingRect(cnt)
                matches.append({"x": x + w/2, "y": y + h/2, "type": "Zwembad Match"})

    if ai_analysis:
        dak_kleur = str(ai_analysis.get("dak_kenmerken", {}).get("kleur", "")).lower()
        if "rood" in dak_kleur or "oranje" in dak_kleur or "pan" in dak_kleur:
            mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
            contours, _ = cv2.findContours(mask1 | mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if 100 < cv2.contourArea(cnt) < 2000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    matches.append({"x": x + w/2, "y": y + h/2, "type": "Dak Match"})
    return matches

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("1. Woningfoto's")
    uploaded_files = st.file_uploader("Upload makelaarsfoto's", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    st.header("2. Locatie Bepalen")
    location_method = st.radio("Kies invoermethode:", ["Plaatsnaam + Kaart", "Link/URL plakken", "Handmatig"])
    
    if location_method == "Link/URL plakken":
        listing_url = st.text_input("Plak de advertentie URL:")
        if listing_url:
            coords = extract_coords_from_url(listing_url)
            if coords:
                st.session_state.search_lat, st.session_state.search_lon = coords
                st.success("Coördinaten ingeladen!")
                
    elif location_method == "Plaatsnaam + Kaart":
        town_input = st.text_input("Plaatsnaam (bijv. Aynac):", value="Aynac")
        if town_input and town_input != st.session_state.last_town:
            geo_coords = geocode_french_town(town_input)
            if geo_coords:
                st.session_state.map_center = [geo_coords[0], geo_coords[1]]
                st.session_state.search_lat = geo_coords[0]
                st.session_state.search_lon = geo_coords[1]
                st.session_state.last_town = town_input
                st.rerun()

    elif location_method == "Handmatig":
        st.session_state.search_lat = st.number_input("Lat", value=st.session_state.search_lat, format="%.6f")
        st.session_state.search_lon = st.number_input("Lon", value=st.session_state.search_lon, format="%.6f")

    grid_size = st.slider("Zoekbereik rondom de pin", min_value=1, max_value=7, value=3, step=2)
    start_search = st.button("Start AI Analyse & Zoekopdracht 🚀", type="primary")

# --- 6. HOOFDWEERGAVE ---

# 6A. Als we nog niet gezocht hebben, of we de resultaten willen resetten
if not start_search and st.session_state.found_hits is None:
    if uploaded_files:
        st.subheader(f"{len(uploaded_files)} foto('s) geüpload")
        cols = st.columns(min(len(uploaded_files), 3))
        for i, photo in enumerate(uploaded_files):
            with cols[i % 3]:
                st.image(photo, caption=f"Foto {i+1}", width='stretch')
        
        st.divider()
        
        if client:
            with st.spinner("🤖 GPT-4o OSINT analyseert foto('s)..."):
                image_bytes_list = [f.getvalue() for f in uploaded_files]
                # Sla de analyse op in de session state, tenzij deze al is gedaan voor deze sessie
                if st.session_state.ai_data is None:
                    st.session_state.ai_data = analyze_photos_with_gpt4o(image_bytes_list)
                
            if st.session_state.ai_data:
                st.subheader("📊 OSINT Geolocatie Profiel")
                st.json(st.session_state.ai_data)
                st.success("Analyse voltooid! Klaar voor de kaart-scan.")
            
    if location_method == "Plaatsnaam + Kaart":
        st.divider()
        st.subheader("📍 Klik op de kaart om je zoek-pin te plaatsen")
        m_select = folium.Map(location=st.session_state.map_center, zoom_start=13)
        folium.Circle(location=[st.session_state.search_lat, st.session_state.search_lon], radius=grid_size * 180, color="red", fill=True, fill_opacity=0.1).add_to(m_select)
        folium.Marker(location=[st.session_state.search_lat, st.session_state.search_lon], icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"), tooltip="Huidig zoekgebied").add_to(m_select)
        map_data = st_folium(m_select, width=1000, height=450, key="selection_map")
        
        if map_data and map_data.get("last_clicked"):
            click_lat = map_data["last_clicked"]["lat"]
            click_lon = map_data["last_clicked"]["lng"]
            if click_lat != st.session_state.search_lat or click_lon != st.session_state.search_lon:
                st.session_state.search_lat = click_lat
                st.session_state.search_lon = click_lon
                st.rerun()

# --- 7. UITVOEREN VAN DE SCAN (Of resultaten tonen als ze in session state staan) ---
if start_search:
    st.divider()
    st.subheader("🔍 IGN Satellietbeelden scannen...")
    
    lat_target = st.session_state.search_lat
    lon_target = st.session_state.search_lon
    
    ZOOM = 17
    center_x, center_y = deg2num(lat_target, lon_target, ZOOM)
    found_hits = []
    offset = grid_size // 2
    
    progress = st.progress(0)
    total_tiles = grid_size * grid_size
    step = 0
    
    # Gebruik de data uit de session state
    ai_data = st.session_state.ai_data

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
            
    # Sla de resultaten op in session state
    st.session_state.found_hits = found_hits
    
# Toon de resultaten als ze aanwezig zijn
if st.session_state.found_hits is not None:
    st.success(f"Scan voltooid! {len(st.session_state.found_hits)} potentiële locaties gevonden in het raster.")
    m_results = folium.Map(location=[st.session_state.search_lat, st.session_state.search_lon], zoom_start=15)
    folium.Circle(location=[st.session_state.search_lat, st.session_state.search_lon], radius=grid_size * 180, color="blue", fill=True, fill_opacity=0.1).add_to(m_results)
    
    for hit in st.session_state.found_hits:
        folium.Marker(
            location=[hit["lat"], hit["lon"]],
            popup=f"Match: {hit['type']}",
            icon=folium.Icon(color="red" if "Zwembad" in hit["type"] else "green", icon="home")
        ).add_to(m_results)
        
    st_folium(m_results, width=1000, height=600)
    
    if st.button("⬅️ Terug naar aanpassen (Reset Scan)"):
        st.session_state.found_hits = None
        st.rerun()
