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
st.set_page_config(page_title="Franse Huizen Geolocation Tool", layout="wide")
st.title("🏡 Franse Huizen Geolocation Tool (Meerdere Foto's & AI)")
st.markdown("Upload buitenfoto's, kies je zoekgebied op de **satellietkaart** en de AI zoekt naar een exacte combinatie van factoren.")

# --- INITIALISATIE SESSION STATE ---
if "search_lat" not in st.session_state: st.session_state.search_lat = 44.891237
if "search_lon" not in st.session_state: st.session_state.search_lon = 1.832689
if "map_center" not in st.session_state: st.session_state.map_center = [44.891237, 1.832689]
if "last_town" not in st.session_state: st.session_state.last_town = ""
if "ai_data" not in st.session_state: st.session_state.ai_data = None
if "found_hits" not in st.session_state: st.session_state.found_hits = None

# --- OpenAI Client setup ---
api_key = st.secrets.get("OPENAI_API_KEY", None)
if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None
    st.warning("⚠️ Geen `OPENAI_API_KEY` gevonden in Streamlit Secrets. AI-fotoanalyse staat uit.")

# --- 2. HULPFUNCTIES ---
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
    Jij bent een strenge OSINT- en cartografie-expert. Genereer een gedetailleerd 'Kavel-DNA'.
    
    Maak een JSON met de volgende velden:
    {
      "dak": {
          "hoofdvorm": "bijv. complex L-vormig, U-vormig, eenvoudig rechthoekig, T-vormig",
          "kleur_signatuur": "bijv. donker leisteen, terracotta rode dakpannen"
      },
      "kavel_blauwdruk": {
          "vegetatie_relatie": "Zeer belangrijk! Staan er bomen strak tegen het huis? Is het een open veld? Waar staan de bomen t.o.v. het gebouw?",
          "bijgebouwen_relatie": "Zijn er permanente bijgebouwen dichtbij zichtbaar?",
          "oprit_en_omgeving": "Is er een zichtbare oprit of weg direct naast het huis?"
      },
      "must_haves_voor_match": "Geef een lijstje van 3 harde eisen waar de satelliet-locatie aan MOET voldoen (bijv. 'Moet bomen aan de westkant hebben EN een rechthoekig dak')."
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

# --- 4. SATELLIET TEGELS ---
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

# --- 5. KEIHARDE AI VERIFICATIE ---
def verify_daken_with_ai(base64_tile, daken_coords_list, kavel_dna_json):
    if not client: return None

    daken_prompt = json.dumps(daken_coords_list)
    kavel_dna_prompt = json.dumps(kavel_dna_json)

    prompt = f"""
    Je ziet een satelliet-tegel met gemarkeerde potentiële daken (pixel-coördinaten [x,y,w,h]).
    Daken lijst: {daken_prompt}
    
    Jij bent de eind-keurmeester. Je MOET een dak AFKEUREN als het niet aan de COMBINATIE van factoren uit het Kavel-DNA voldoet.
    Kavel-DNA: {kavel_dna_prompt}
    
    REGELS:
    1. Een overeenkomende dakkleur is ONVOLDOENDE.
    2. Kijk kritisch naar de 'vegetatie_relatie' en 'must_haves_voor_match'. Als de Kavel-DNA zegt dat er bomen vlakbij staan, en dit dak ligt in een open veld, dan is het GEEN match.
    3. Liever 0 matches dan onbetrouwbare matches. Geef alleen 'hoog' als vorm, kleur, bomen en omgeving allemaal kloppen.
    
    Retourneer:
    {{
      "verified_daken": [
          {{
              "dak_id": 0,
              "match_waarschijnlijkheid": "laag of hoog",
              "redenering": "Waarom wel of niet? Benoem expliciet de vegetatie en vorm.",
              "x_center": 123, "y_center": 456
          }}
      ]
    }}
    """

    messages_content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_tile}"}}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": messages_content}],
            response_format={"type": "json_object"},
            max_tokens=800
        )
        result_json = json.loads(response.choices[0].message.content)
        
        verified_results = []
        for verified_dak in result_json.get("verified_daken", []):
            if verified_dak.get("match_waarschijnlijkheid") == "hoog":
                verified_results.append({
                    "x_tile": verified_dak["x_center"],
                    "y_tile": verified_dak["y_center"],
                    "type": f"Exacte Match: {verified_dak['redenering']}"
                })
        return verified_results
    except Exception as e:
        return None

# --- MAP HULPFUNCTIE MET SATELLIET ---
def create_satellite_map(lat, lon, zoom):
    """Maakt een Folium map met Franse IGN satellietbeelden als basislaag."""
    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles=None)
    folium.TileLayer(
        tiles='https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=ORTHOIMAGERY.ORTHOPHOTOS&STYLE=normal&FORMAT=image/jpeg&TILEMATRIXSET=PM&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}',
        attr='IGN Frankrijk',
        name='IGN Satelliet',
        overlay=False,
        control=True
    ).add_to(m)
    return m

# --- 6. SIDEBAR ---
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

    grid_size = st.slider("Zoekbereik rondom de pin (aantal tegels)", min_value=1, max_value=7, value=3, step=2)
    start_search = st.button("Start Strenge AI Zoekopdracht 🚀", type="primary")

# --- 7. HOOFDWEERGAVE (VOOR HET ZOEKEN) ---
if not start_search and st.session_state.found_hits is None:
    if uploaded_files:
        st.subheader(f"{len(uploaded_files)} foto('s) geüpload")
        cols = st.columns(min(len(uploaded_files), 3))
        for i, photo in enumerate(uploaded_files):
            with cols[i % 3]:
                st.image(photo, caption=f"Foto {i+1}", width='stretch')
        
        st.divider()
        
        if client:
            with st.spinner("🤖 GPT-4o berekent strikt Kavel-DNA..."):
                image_bytes_list = [f.getvalue() for f in uploaded_files]
                if st.session_state.ai_data is None:
                    st.session_state.ai_data = analyze_photos_with_gpt4o(image_bytes_list)
                
            if st.session_state.ai_data:
                st.subheader("📊 Strikte Woning Blauwdruk")
                st.json(st.session_state.ai_data)
            
    if location_method == "Plaatsnaam + Kaart":
        st.divider()
        st.subheader("📍 Klik op de satellietkaart om je exacte zoek-pin te plaatsen")
        
        m_select = create_satellite_map(st.session_state.map_center[0], st.session_state.map_center[1], 14)
        folium.Circle(location=[st.session_state.search_lat, st.session_state.search_lon], radius=grid_size * 180, color="red", fill=True, fill_opacity=0.3).add_to(m_select)
        folium.Marker(location=[st.session_state.search_lat, st.session_state.search_lon], icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"), tooltip="Zoekmiddelpunt").add_to(m_select)
        map_data = st_folium(m_select, width=1000, height=450, key="selection_map")
        
        if map_data and map_data.get("last_clicked"):
            click_lat = map_data["last_clicked"]["lat"]
            click_lon = map_data["last_clicked"]["lng"]
            if click_lat != st.session_state.search_lat or click_lon != st.session_state.search_lon:
                st.session_state.search_lat = click_lat
                st.session_state.search_lon = click_lon
                st.rerun()

# --- 8. UITVOEREN VAN DE SCAN ---
if start_search:
    st.divider()
    st.subheader("🔍 IGN Satellietbeelden scannen (Strenge Modus)...")
    
    lat_target = st.session_state.search_lat
    lon_target = st.session_state.search_lon
    ZOOM = 17
    center_x, center_y = deg2num(lat_target, lon_target, ZOOM)
    offset = grid_size // 2
    
    progress = st.progress(0)
    total_tiles = grid_size * grid_size
    step = 0
    kavel_dna = st.session_state.ai_data
    st.session_state.found_hits = []

    for dx in range(-offset, offset + 1):
        for dy in range(-offset, offset + 1):
            tx = center_x + dx
            ty = center_y + dy
            tile_img = fetch_ign_satellite_tile(tx, ty, ZOOM)
            
            if tile_img and client:
                img_np = np.array(tile_img)
                hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                potentiële_daken = []

                # Zoek alle gebouwen (breder spectrum: rode en donkere daken)
                mask1 = cv2.inRange(hsv, np.array([0, 50, 20]), np.array([20, 255, 255]))
                mask2 = cv2.inRange(hsv, np.array([160, 50, 20]), np.array([180, 255, 255]))
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                _, mask3 = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV) # Donkere leisteen daken
                
                contours, _ = cv2.findContours(mask1 | mask2 | mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if 100 < area < 3000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        potentiële_daken.append([x, y, w, h])
                
                if potentiële_daken:
                    buffered = io.BytesIO()
                    tile_img.save(buffered, format="JPEG")
                    base64_tile = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    verified_results = verify_daken_with_ai(base64_tile, potentiële_daken, kavel_dna)
                    if verified_results:
                        nw_lat, nw_lon = num2deg(tx, ty, ZOOM)
                        se_lat, se_lon = num2deg(tx + 1, ty + 1, ZOOM)
                        for res in verified_results:
                            hit_lon = nw_lon + (res["x_tile"] / 256.0) * (se_lon - nw_lon)
                            hit_lat = nw_lat + (res["y_tile"] / 256.0) * (se_lat - nw_lat)
                            st.session_state.found_hits.append({
                                "lat": hit_lat, "lon": hit_lon, 
                                "type": res["type"]
                            })
            
            step += 1
            progress.progress(step / total_tiles)
            
# --- 9. RESULTATEN WEERGAVE (MET SATELLIETKAART) ---
if st.session_state.found_hits is not None:
    st.divider()
    if len(st.session_state.found_hits) > 0:
        st.success(f"Scan voltooid! {len(st.session_state.found_hits)} harde match(es) gevonden die aan alle voorwaarden voldoen.")
    else:
        st.warning("Scan voltooid. 0 matches gevonden. De AI heeft alle daken afgekeurd omdat de combinatie (bijv. vegetatie of vorm) niet overeenkwam met de foto's.")

    m_results = create_satellite_map(st.session_state.search_lat, st.session_state.search_lon, 16)
    folium.Circle(location=[st.session_state.search_lat, st.session_state.search_lon], radius=grid_size * 180, color="blue", fill=False).add_to(m_results)
    
    for hit in st.session_state.found_hits:
        folium.Marker(
            location=[hit["lat"], hit["lon"]],
            popup=f"Match: {hit['type']}",
            icon=folium.Icon(color="green", icon="check", prefix="fa") 
        ).add_to(m_results)
        
    st_folium(m_results, width=1000, height=600)
    
    if st.button("⬅️ Terug naar aanpassen (Reset Scan)"):
        st.session_state.found_hits = None
        st.rerun()
