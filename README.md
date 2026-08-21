# HouseFinder v2

Een kostenbegrensde zoektool voor Franse woningadvertenties. De applicatie vindt eerst **alle gebouwcontouren binnen de echte zoekcirkel** met open Franse geo-data. Daarna worden de beste kandidaten lokaal gescoord en in maximaal twee compacte vision-calls vergeleken.

## Wat er fundamenteel is veranderd

- Geen AI-call meer per kaarttegel of gebouw.
- Exacte cirkelfiltering in plaats van een afwijkend vierkant raster.
- Kandidaatgeneratie via de actuele IGN-laag `BDTOPO_V3:batiment`.
- Hoge-resolutie IGN-crops op zoomniveau 19, alleen voor de lokale shortlist.
- Rode gebouwcontour op iedere crop, zodat het model weet welk dak wordt beoordeeld.
- Oorspronkelijke woningfoto's blijven beschikbaar voor de vergelijking; er is geen tekst-only bottleneck.
- `gpt-5.6-luna` is het goedkope standaardmodel met Structured Outputs.
- `gpt-5.6-terra` wordt alleen gebruikt wanneer de topresultaten ambigu zijn.
- Voor iedere modelcall wordt vooraf het maximale bedrag berekend. Een call die het ingestelde budget overschrijdt wordt niet uitgevoerd.
- Zonder OpenAI-sleutel blijft de applicatie bruikbaar met handmatige aanwijzingen en lokale geometrische ranking.

## Installeren op Windows

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .streamlit\secrets.example.toml .streamlit\secrets.toml
```

Vul daarna optioneel je API-sleutel in `.streamlit/secrets.toml` in en start de app:

```powershell
streamlit run app.py
```

De sleutel kan ook via de omgevingsvariabele `OPENAI_API_KEY` worden aangeboden.

## Werkwijze

1. Upload maximaal zes informatieve buitenfoto's.
2. Laat Luna de vanuit de lucht bruikbare aanwijzingen extraheren of vul ze handmatig in.
3. Controleer de aanwijzingen. Afwezigheid van een bijgebouw is standaard **geen** harde conclusie.
4. Vul het midden en de straal van de echte makelaarscirkel in of klik het midden op de kaart aan.
5. Start de zoekopdracht.
6. Controleer de gerangschikte kandidaten, tegenstrijdigheden en onzekerheden handmatig.

## Kostenbeheersing

De standaardlimiet is `$0.04` per woning. De kostenmodule telt 32×32-beeldpatches, reserveert vóór een call het maximale outputbudget en verwerkt na afloop de werkelijke `usage` uit de API-respons. De gebruikersinterface toont een kostenlog per stap.

Normaal pad:

- één Luna-call voor fotoanalyse;
- lokale kandidaatgeneratie en scoring zonder AI-kosten;
- één Luna-call met maximaal vijftien kandidaten in één contact sheet;
- alleen bij ambiguïteit één Terra-call met maximaal vier kandidaten.

Modelprijzen staan centraal in `housefinder/config.py`. Controleer ze periodiek tegen de officiële OpenAI-prijspagina voordat je dit als product aanbiedt.

## Gratis databronnen

- IGN BD TOPO via WFS voor gebouwcontouren;
- IGN BD ORTHO via WMTS voor luchtfoto's;
- de Franse Géoplateforme/BAN-geocoder;
- Folium/Leaflet voor de kaart.

De openbare IGN-data vallen onder de Licence Ouverte Etalab. Respecteer desondanks fair-use, attributie en eventuele actuele servicelimieten. Tegels en WFS-resultaten worden lokaal gecachet om de publieke diensten zo min mogelijk te belasten.

## Projectstructuur

```text
app.py                     Streamlit-interface
housefinder/ai.py          budgetbegrensde Responses API-calls
housefinder/costs.py       token- en kostenbewaking
housefinder/geo.py         cirkels, URL's en Web-Mercator
housefinder/ign.py         WFS-kandidaten en geocoding
housefinder/imagery.py     WMTS-cache, crops en contact sheets
housefinder/scoring.py     lokale, uitlegbare ranking
housefinder/pipeline.py    georkestreerde zoekopdracht
tests/                     offline unit-tests
```

## Testen

```powershell
python -m pip install -r requirements-dev.txt
pytest
ruff check .
```

De normale tests doen geen betaalde modelcalls en gebruiken geen live IGN-netwerkverkeer.

## Belangrijke beperkingen

- Een hoge score is een kandidaat, geen bewijs van de exacte locatie.
- Luchtfoto's en advertentiefoto's kunnen jaren uit elkaar liggen.
- Kleine of zeer recente gebouwen kunnen nog ontbreken in BD TOPO.
- Bomen, schaduwen en seizoenen kunnen objecten verbergen.
- De geometrische score gebruikt gebouwcontouren; vegetatie, oprit en dakkleur worden pas in de vision-reranking beoordeeld.
