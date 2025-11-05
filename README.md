# Photic Driving Response
Geautomatiseerde pipeline voor het detecteren en kwantificeren van biomarkers van de volgreactie in EEGs.

## 1 Installatie
Om de repository te gebruiken heb je volgende vereisten nodig:

1. Python versie 3.13.1 of hoger
2. De repository via https://github.com/livkrom/PhoticDrivingResponse.git
    - Werk vanuit de folder die je via stap 3 verkrijgt
3. Alle dependencies vanuit de requirements.txt 

## 2 Gebruik
Alle scripts worden aangestuurd via main. Deze kan aangestuurd worden via Visual Studio Code (of een andere environment) of via je terminal. Overigens kan klad worden verwijderd, dit script werd enkel gebruikt om aparte functies te testen. Alle gemaakte resultaten komen in dezelfde GitHub folder onder 'results', voor elke complete run mag dit worden verwijderd. Onder elke functie staat er bij wat het doet en hoe het werkt. Voor theoretische achtergrond van verwerkingsstappen, raadpleeg het eindrapport. 

### 2.1 Patient files
In de module 'patients' staan de bijbehorende functies voor omgang met patiënt data. 
1. De code gaat nu uit van verschillende mappen waarin de patiënt-data gesorteerd zijn, maar dit kan in principe genegeerd worden. Afhankelijk van welke subgroepen je wilt analyseren moeten de defaults onder 'parse_args' aangepast worden. 
2. Vervolgens wordt alle .cnt data uit de folder gezocht die onder 'data_folder' in de functie patient_files staat, deze moet nog worden aangepast naar waar de data staat. 
3. Er wordt er nu van uit gegaan dat alles onder VEPXX_T.cnt vermeld staat; XX staat voor het patiëntnummer en T voor het tijdstip (1 --> t0, 2 --> t1, 3 --> t2)

### 2.2 Biomarkers
Per patiënt vanuit 'data_folder' uit 2.1.2 worden verwerkt
1. De power-gerelateerde biomarkers worden geheel uit de 'power' module verwerkt.
2. De PLV biomarker wordt geheel uit de 'phase' module verwerkt. 

Alleen als er bij een patient alle biomarkers voor alle tijdspuntn succesvol zijn verwerkt, wordt dit gezien als een complete set. Complete sets worden doorgezet naar 'results_PLV' en 'results_POWER'. Incomplete sets worden doorgezet naar 'results_incomplete'.

### 2.3 Statistiek & Classificatie
1. Vanuit alle complete sets wordt de statistiek gedaan via de module 'statistics'. De tabellen die hieruit komen als resultaten worden opgeslagen onder 'results'.  

2. Vanuit alle complete sets wordt de classificatie gedaan via de module 'classification'. De resultaten hiervan worden geprint in de terminal. 