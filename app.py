# --- IMPORTS ---
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

# --- 1. CONFIGURATIE & UI (Fixuse_container_width -> width='stretch') ---
st.set_page_config(page_title="Franse Huizen Geolocation Tool (Meerdere Foto's & AI)", layout="wide")
st.title("🏡 Franse Huizen Geolocation Tool (Meerdere Foto's & AI)")
st.markdown("Upload meerdere buitenfoto's, kies je zoekgebied op de interactieve kaart en laat de AI scannen.")

# --- INITIALISATIE SESSION STATE (uitgebreid) ---
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
            return coords[1], coords[0] # Lat, Lon
    except Exception: pass
    return None

# --- 3. AI FOTO-ANALYSE FUNCTIE (HERZIEN - GENERATE KAVEL DNA) ---
def analyze_photos_with_gpt4o(image_bytes_list):
    if not client or not image_bytes_list: return None
    
    # HERZIENE PROMPT: Generate a 'Kavel-DNA' blauwdruk
    prompt = """
    Jij bent een OSINT- en cartografie-expert. Jouw doel is om een 'Kavel-DNA' te genereren 
    van de woning op de foto's, die we kunnen vergelijken met satellietbeelden.
    
    Maak een JSON met de volgende velden:
    {
      "dak": {
          "hoofdvorm": "bijv. complex L-vormig, U-vormig, eenvoudig rechthoekig, T-vormig",
          "kleur_signatuur": "bijv. donker leisteen, terracotta rode dakpannen",
          "opvallende_kenmerken": "bijv. drie dakkapellen op het zuidwesten, een schoorsteen op het midden (indien zichtbaar)"
      },
      "kavel_blauwdruk": {
          "hoofdgebouw_oriëntatie": "Beschrijf hoe het hoofdgebouw staat t.o.v. de (vermoedelijke) weg. Bijv. 'hoofdas parallel aan de weg'",
          "oprit_locatie": "bijv. onverhard pad aan de oostzijde, geasfalteerde oprit in het midden",
          "bijgebouwen_relatie": "Beschrijf permanente bijgebouwen (bijv. schuur) ten opzichte van het hoofdgebouw. Bijv. 'kleine schuur direct ten noorden van de kavel'",
          "vegetatie_relatie": "Beschrijf opvallende vegetatie t.o.v. het dak. Bijv. 'grote loofboom overdekt de noordoostelijke hoek van het dak'"
      },
      "harde_markers": {
          "is_zwembad_permanent": true/false, // Alleen permanent, ingegraven zwembad
          "muren_hekwerk": "Is er een opvallende muur of hekwerk aan de straatkant?"
      }
    }
    """
    
    messages_content = [{"type": "text", "text": prompt}]
    for img_bytes in image_bytes_list:
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        messages_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })
        
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

# --- 5. HERZIENE VERIFICATIE FUNCTIES (INTELLIGENT SCANNEN) ---

def verify_daken_with_ai(base64_tile, daken_coords_list, kavel_dna_json):
    """
    Maakt een verificatie API-call naar GPT-4o Vision voor een lijst met potentiële daken op een tegel.
    """
    if not client: return None

    # We moeten de daken-coördinaten in de prompt opnemen
    daken_prompt = json.dumps(daken_coords_list)
    kavel_dna_prompt = json.dumps(kavel_dna_json)

    prompt = f"""
    Je ziet een satelliet-tegel met een lijst van potentiële daken, gemarkeerd door hun pixel-coördinaten [x,y,w,h] (top-left).
    Daken lijst: {daken_prompt}
    
    Vergelijk deze daken en hun directe omgeving met het 'Kavel-DNA' van de gezochte woning.
    Kavel-DNA: {kavel_dna_prompt}
    
    Retourneer een JSON met een match-score voor elk dak. Alleen daken met 'match_waarschijnlijkheid': 'hoog' worden geaccepteerd.
    JSON structuur:
    {
      "verified_daken": [
          {{
              "dak_id": 0, // index in de daken lijst
              "match_waarschijnlijkheid": "bijv. laag, medium, hoog",
              "redenering": "bijv. 'vorm komt overeen en de oprit is correct gepositioneerd'",
              "x_center": 123, "y_center": 456 // de center coördinaten van het match-dak op de tegel (0-256)
          }},
          // ...
      ]
    }
    """

    messages_content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_tile}"}}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o", # We gebruiken gpt-4o voor de intelligentie, ook al is het duurder.
            messages=[{"role": "user", "content": messages_content}],
            response_format={"type": "json_object"},
            max_tokens=800
        )
        result_json = json.loads(response.choices[0].message.content)
        
        # Filter alleen de 'hoog' matches en formatteer ze voor de kaart
        verified_results = []
        for verified_dak in result_json.get("verified_daken", []):
            if verified_dak.get("match_waarschijnlijkheid") == "hoog":
                verified_results.append({
                    "x_tile": verified_dak["x_center"], # Pixel coordinaat op de tegel
                    "y_tile": verified_dak["y_center"],
                    "type": f"AI Geverifieerd: {verified_dak['redenering']}"
                })
        return verified_results
    except Exception as e:
        st.error(f"Fout bij AI verificatie: {e}")
        return None

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

    grid_size = st.slider("Zoekbereik rondom de pin", min_value=1, max_value=7, value=3, step=2)
    start_search = st.button("Start Intelligent Analyse & Zoekopdracht 🚀", type="primary")

# --- 7. HOOFDWEERGAVE ---

# 6A. Als we nog niet gezocht hebben
if not start_search and st.session_state.found_hits is None:
    if uploaded_files:
        st.subheader(f"{len(uploaded_files)} foto('s) geüpload")
        cols = st.columns(min(len(uploaded_files), 3))
        for i, photo in enumerate(uploaded_files):
            with cols[i % 3]:
                # Fix use_container_width -> width='stretch'
                st.image(photo, caption=f"Foto {i+1}", width='stretch')
        
        st.divider()
        
        # AI Analyse: Genereer 'Kavel-DNA'
        if client:
            with st.spinner("🤖 GPT-4o genereert Kavel-DNA van de woning..."):
                image_bytes_list = [f.getvalue() for f in uploaded_files]
                # Sla de analyse op in de session state, tenzij deze al is gedaan
                if st.session_state.ai_data is None:
                    st.session_state.ai_data = analyze_photos_with_gpt4o(image_bytes_list)
                
            if st.session_state.ai_data:
                st.subheader("📊 Woning Blauwdruk (Kavel-DNA)")
                st.json(st.session_state.ai_data)
                st.success("DNA geanalyseerd! Klaar voor het intelligent scannen van de kaart.")
            
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

# --- 8. UITVOEREN VAN DE INTELLIGENTE SCAN ---
if start_search:
    st.divider()
    st.subheader("🔍 IGN Satellietbeelden intelligent scannen en verifiëren...")
    
    lat_target = st.session_state.search_lat
    lon_target = st.session_state.search_lon
    
    ZOOM = 17
    center_x, center_y = deg2num(lat_target, lon_target, ZOOM)
    offset = grid_size // 2
    
    progress = st.progress(0)
    total_tiles = grid_size * grid_size
    step = 0
    
    # Gebruik de data uit de session state (het Kavel-DNA)
    kavel_dna = st.session_state.ai_data
    
    # We initialiseren found_hits als een lege lijst in session state
    st.session_state.found_hits = []

    # De loop is nu intelligent: Download tegel -> Vind potentiële daken -> Verifieer met AI
    for dx in range(-offset, offset + 1):
        for dy in range(-offset, offset + 1):
            tx = center_x + dx
            ty = center_y + dy
            tile_img = fetch_ign_satellite_tile(tx, ty, ZOOM)
            
            if tile_img and client:
                # 1. Grof filteren: Vind potentiële daken met OpenCV (Rood)
                # We moeten de OpenCV logica hier integreren om een daken-coördinaten lijst te maken.
                img_np = np.array(tile_img)
                hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                potentiële_daken = []

                # Rode daken filter
                mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
                mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
                contours, _ = cv2.findContours(mask1 | mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if 100 < area < 2000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        # Voeg potentieel dak toe (pixel coordinaat [x,y,w,h])
                        potentiële_daken.append([x, y, w, h])
                
                # 2. Zwembaden filteren (Optioneel, als harde marker, maar de intelligent scan is hoofdzaak)
                # ...

                # 3. Verificatie-fase: Stuur de daken en de tegel terug naar de AI
                if potentiële_daken:
                    # Converteer de tile naar base64 voor de API
                    buffered = io.BytesIO()
                    tile_img.save(buffered, format="JPEG")
                    base64_tile = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                    verified_results = verify_daken_with_ai(base64_tile, potentiële_daken, kavel_dna)
                    if verified_results:
                        # Zet verified pixel coordinaten om naar GPS coordinaten en voeg toe aan hits
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
            
# Toon de resultaten als ze aanwezig zijn
if st.session_state.found_hits is not None:
    st.divider()
    if len(st.session_state.found_hits) > 0:
        st.success(f"Intelligente scan voltooid! {len(st.session_state.found_hits)} locaties gevonden die hoog scoren op het Kavel-DNA.")
    else:
        st.warning("Intelligente scan voltooid. Geen locaties gevonden die hoog scoorden op het Kavel-DNA. Probeer de zoek-pin te verplaatsen of het zoekbereik te vergroten.")

    # Fixuse_container_width -> width='stretch' in de kaartweergave
    m_results = folium.Map(location=[st.session_state.search_lat, st.session_state.search_lon], zoom_start=15)
    folium.Circle(location=[st.session_state.search_lat, st.session_state.search_lon], radius=grid_size * 180, color="blue", fill=True, fill_opacity=0.1).add_to(m_results)
    
    for hit in st.session_state.found_hits:
        folium.Marker(
            location=[hit["lat"], hit["lon"]],
            popup=f"Match: {hit['type']}",
            # Gebruik een andere icon voor verified matches
            icon=folium.Icon(color="green", icon="verified", prefix="fa") 
        ).add_to(m_results)
        
    st_folium(m_results, width=1000, height=600)
    
    if st.button("⬅️ Terug naar aanpassen (Reset Scan)"):
        # Reset de gevonden hits, maar houd de AI-data en kaartpositie vast
        st.session_state.found_hits = None
        st.rerun()
