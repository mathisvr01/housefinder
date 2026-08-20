import streamlit as st
import requests
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
st.set_page_config(page_title="Franse Huizen OSINT Tool", layout="wide")
st.title("🏡 Franse Huizen OSINT Tool (Strenge Deep Scan)")
st.markdown("Upload buitenfoto's en kies je zoekgebied. De AI filtert streng op grootte, bijgebouwen en vegetatie. Klik op resultaten voor **Street View**.")

# --- INITIALISATIE SESSION STATE ---
for key, default in [("search_lat", 44.891237), ("search_lon", 1.832689), 
                     ("map_center", [44.891237, 1.832689]), ("last_town", ""), 
                     ("ai_data", None), ("found_hits", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- OpenAI Client setup ---
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
except:
    client = None
    st.warning("⚠️ Geen `OPENAI_API_KEY` gevonden in Streamlit Secrets. AI-fotoanalyse staat uit.")

# --- 2. HULPFUNCTIES LOCATIE ---
def extract_coords_from_url(url: str):
    m_bienici = re.search(r'camera=\d+_([0-9.-]+)_([0-9.-]+)', url)
    if m_bienici: return float(m_bienici.group(2)), float(m_bienici.group(1))
    m_gmaps = re.search(r'@([0-9.-]+),([0-9.-]+)', url)
    if m_gmaps: return float(m_gmaps.group(1)), float(m_gmaps.group(2))
    return None

def geocode_french_town(town_name: str):
    url = f"https://api-adresse.data.gouv.fr/search/?q={town_name}&limit=1"
    try:
        res = requests.get(url, timeout=5).json()
        if res.get('features'):
            coords = res['features'][0]['geometry']['coordinates']
            return coords[1], coords[0]
    except: pass
    return None

# --- 3. AI FOTO-ANALYSE FUNCTIE ---
def analyze_photos_with_gpt4o(image_bytes_list):
    if not client or not image_bytes_list: return None
    
    prompt = """
    Jij bent een meedogenloze OSINT-expert. Maak een 'Kavel-DNA' van dit huis.
    Gebruik GEEN windrichtingen (zoals Noord/Zuid), maar relatieve posities ('naast het huis', 'achter de oprit').
    
    Maak een JSON:
    {
      "dak_en_grootte": "Vorm en kleur van het dak. Schat de relatieve grootte van de voetafdruk in (klein huisje, grote boerderij, langwerpig, etc.).",
      "kavel_context": {
          "vegetatie": "Staan er bomen strak tegen het huis? Of in open veld? Waar precies?",
          "bijgebouwen_relatie": "Zijn er bijgebouwen? ZO NEE, vermeld expliciet 'GEEN BIJGEBOUWEN' zodat we boerderijcomplexen kunnen afkeuren.",
          "oprit_en_infrastructuur": "Is er een zichtbare oprit of weg?"
      },
      "strikte_combinatie_eis": "Wat is de harde eis? (bijv. Moet losstaand zijn zonder schuren, mét bomen aan de achterkant)"
    }
    """
    
    messages = [{"type": "text", "text": prompt}]
    for img_bytes in image_bytes_list:
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        messages.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
        
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": messages}],
            response_format={"type": "json_object"},
            max_tokens=500
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Fout bij AI analyse: {e}")
        return None

# --- 4. SATELLIET TEGELS OPHALEN ---
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

# --- 5. STRENGE DEEP SCAN AI VERIFICATIE ---
def deep_scan_tile_with_ai(base64_tile, kavel_dna_json):
    if not client: return None

    prompt = f"""
    Jij bent een keiharde OSINT satelliet-expert. Scan deze hele satelliet-tegel.
    Kavel-DNA profiel waarnaar we zoeken:
    {json.dumps(kavel_dna_json)}
    
    BEOORDEEL STRENG OP DE VOLGENDE FACTOREN:
    1. GROOTTE & VORM: Komt de voetafdruk/maat van het gebouw overeen?
    2. BIJGEBOUWEN: Als het DNA zegt 'geen bijgebouwen', KEUR DAN elk boerderijcomplex met meerdere daken AF (score: laag).
    3. VEGETATIE: Als het DNA bomen eist rondom het huis, keur dan huizen in een kaal weiland AF.
    
    Als je een gebouw vindt dat een écht goede kandidaat is, schat de positie in percentages (X=links naar rechts, Y=boven naar beneden).
    
    Retourneer een JSON:
    {{
      "matches": [
          {{
              "score": "hoog (uitstekende match) of medium (redelijke twijfelgeval, sluit niets uit)",
              "redenering": "Verklaar waarom dit klopt qua maat, bomen en bijgebouwen.",
              "x_percentage": 50,
              "y_percentage": 50
          }}
      ]
    }}
    """

    messages = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_tile}"}}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": messages}],
            response_format={"type": "json_object"},
            max_tokens=800
        )
        result_json = json.loads(response.choices[0].message.content)
        
        valid_matches = []
        for match in result_json.get("matches", []):
            score = str(match.get("score", "laag")).lower()
            if "hoog" in score or "medium" in score:
                valid_matches.append({
                    "x_perc": match.get("x_percentage", 50),
                    "y_perc": match.get("y_percentage", 50),
                    "score": "hoog" if "hoog" in score else "medium",
                    "type": f"{match.get('redenering', '')}"
                })
        return valid_matches
    except:
        return None

# --- SATELLIET KAART HULPFUNCTIE ---
def create_satellite_map(lat, lon, zoom):
    m = folium.Map(location=[lat, lon], zoom_start=zoom, tiles=None)
    folium.TileLayer(
        tiles='https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0&LAYER=ORTHOIMAGERY.ORTHOPHOTOS&STYLE=normal&FORMAT=image/jpeg&TILEMATRIXSET=PM&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}',
        attr='IGN Frankrijk', name='IGN Satelliet', overlay=False, control=True
    ).add_to(m)
    return m

# --- 6. SIDEBAR ---
with st.sidebar:
    st.header("1. Woningfoto's")
    uploaded_files = st.file_uploader("Upload foto's", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    st.header("2. Locatie Bepalen")
    location_method = st.radio("Kies methode:", ["Plaatsnaam + Kaart", "Link plakken", "Handmatig Lat/Lon"])
    
    if location_method == "Link plakken":
        listing_url = st.text_input("Plak de URL (Bien'ici/Google Maps):")
        if listing_url:
            coords = extract_coords_from_url(listing_url)
            if coords:
                st.session_state.search_lat, st.session_state.search_lon = coords
                st.success("Coördinaten ingeladen!")
                
    elif location_method == "Plaatsnaam + Kaart":
        town_input = st.text_input("Plaatsnaam:", value="Aynac")
        if town_input and town_input != st.session_state.last_town:
            geo_coords = geocode_french_town(town_input)
            if geo_coords:
                st.session_state.map_center = [geo_coords[0], geo_coords[1]]
                st.session_state.search_lat = geo_coords[0]
                st.session_state.search_lon = geo_coords[1]
                st.session_state.last_town = town_input
                st.rerun()

    elif location_method == "Handmatig Lat/Lon":
        st.session_state.search_lat = st.number_input("Lat", value=st.session_state.search_lat, format="%.6f")
        st.session_state.search_lon = st.number_input("Lon", value=st.session_state.search_lon, format="%.6f")

    grid_size = st.slider("Zoekbereik (aantal tegels)", min_value=1, max_value=7, value=3, step=2)
    start_search = st.button("Start Strenge AI Scan 🚀", type="primary")

# --- 7. HOOFDWEERGAVE ---
if not start_search and st.session_state.found_hits is None:
    if uploaded_files:
        cols = st.columns(min(len(uploaded_files), 3))
        for i, photo in enumerate(uploaded_files):
            with cols[i % 3]:
                st.image(photo, width='stretch')
        
        st.divider()
        if client:
            with st.spinner("🤖 Kavel-DNA berekenen..."):
                if st.session_state.ai_data is None:
                    image_bytes_list = [f.getvalue() for f in uploaded_files]
                    st.session_state.ai_data = analyze_photos_with_gpt4o(image_bytes_list)
            if st.session_state.ai_data:
                st.subheader("📊 Strikte Woning Blauwdruk")
                st.json(st.session_state.ai_data)
            
    if location_method == "Plaatsnaam + Kaart":
        st.divider()
        st.subheader("📍 Plaats je zoek-pin")
        m_select = folium.Map(location=st.session_state.map_center, zoom_start=13)
        folium.Circle(location=[st.session_state.search_lat, st.session_state.search_lon], radius=grid_size * 180, color="red", fill=True, fill_opacity=0.3).add_to(m_select)
        folium.Marker(location=[st.session_state.search_lat, st.session_state.search_lon], icon=folium.Icon(color="red")).add_to(m_select)
        map_data = st_folium(m_select, width=1000, height=450, key="selection_map")
        
        if map_data and map_data.get("last_clicked"):
            click_lat = map_data["last_clicked"]["lat"]
            click_lon = map_data["last_clicked"]["lng"]
            if click_lat != st.session_state.search_lat or click_lon != st.session_state.search_lon:
                st.session_state.search_lat = click_lat
                st.session_state.search_lon = click_lon
                st.rerun()

# --- 8. SCAN UITVOEREN ---
if start_search:
    st.divider()
    st.subheader("🔍 Deep Scan Bezig met OSINT-regels...")
    
    lat_target = st.session_state.search_lat
    lon_target = st.session_state.search_lon
    ZOOM = 17
    center_x, center_y = deg2num(lat_target, lon_target, ZOOM)
    offset = grid_size // 2
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_tiles = grid_size * grid_size
    step = 0
    kavel_dna = st.session_state.ai_data
    st.session_state.found_hits = []

    for dx in range(-offset, offset + 1):
        for dy in range(-offset, offset + 1):
            step += 1
            status_text.text(f"Tegel {step} van {total_tiles} streng beoordelen met AI. Even geduld...")
            progress_bar.progress(step / total_tiles)
            
            tx = center_x + dx
            ty = center_y + dy
            tile_img = fetch_ign_satellite_tile(tx, ty, ZOOM)
            
            if tile_img and client:
                buffered = io.BytesIO()
                tile_img.save(buffered, format="JPEG")
                base64_tile = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                matches = deep_scan_tile_with_ai(base64_tile, kavel_dna)
                
                if matches:
                    nw_lat, nw_lon = num2deg(tx, ty, ZOOM)
                    se_lat, se_lon = num2deg(tx + 1, ty + 1, ZOOM)
                    
                    for match in matches:
                        hit_lon = nw_lon + (match["x_perc"] / 100.0) * (se_lon - nw_lon)
                        hit_lat = nw_lat - (match["y_perc"] / 100.0) * (nw_lat - se_lat)
                        st.session_state.found_hits.append({
                            "lat": hit_lat, "lon": hit_lon, 
                            "type": match["type"],
                            "score": match["score"]
                        })
            
    status_text.text("Scan Voltooid!")
            
# --- 9. RESULTATEN TONEN MET GOOGLE STREET VIEW LINKS ---
if st.session_state.found_hits is not None:
    st.divider()
    hits = st.session_state.found_hits
    if len(hits) > 0:
        st.success(f"Deep Scan afgerond! {len(hits)} strenge kandidaten gevonden.")
    else:
        st.warning("Geen kandidaten gevonden. De AI heeft elk huis afgekeurd o.b.v. vorm, bijgebouwen of vegetatie.")

    m_results = create_satellite_map(st.session_state.search_lat, st.session_state.search_lon, 16)
    folium.Circle(location=[st.session_state.search_lat, st.session_state.search_lon], radius=grid_size * 180, color="blue", fill=False).add_to(m_results)
    
    for hit in hits:
        color = "green" if hit["score"] == "hoog" else "orange"
        
        # Hyperlinks genereren voor Google Maps & Street View
        maps_link = f"https://www.google.com/maps/search/?api=1&query={hit['lat']},{hit['lon']}"
        streetview_link = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={hit['lat']},{hit['lon']}"
        
        # HTML Pop-up
        popup_html = f"""
        <div style="font-family: Arial; font-size: 14px; min-width: 220px;">
            <b style="color: {'green' if color == 'green' else '#d97706'};">Score: {hit['score'].upper()}</b><br>
            <p style="margin-top: 5px; margin-bottom: 12px; font-size: 12px;">{hit['type']}</p>
            <a href="{maps_link}" target="_blank" style="display:block; margin-bottom: 5px; text-decoration: none; color: #1a73e8;">🗺️ Open in Google Maps</a>
            <a href="{streetview_link}" target="_blank" style="display:block; text-decoration: none; color: #1a73e8;">🚗 Open in Street View</a>
        </div>
        """
        
        folium.Marker(
            location=[hit["lat"], hit["lon"]],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=color, icon="check" if color=="green" else "search", prefix="fa") 
        ).add_to(m_results)
        
    st_folium(m_results, width=1000, height=600)
    
    if st.button("⬅️ Terug naar aanpassen"):
        st.session_state.found_hits = None
        st.rerun()
