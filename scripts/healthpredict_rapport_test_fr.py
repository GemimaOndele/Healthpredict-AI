# Create a bilingual test dataset (FR/EN) of maintenance-report texts for HealthPredict AI
import pandas as pd
import numpy as np

# Helper: frequency by equipment type
freq_by_type = {
    "Défibrillateur": 1000,
    "Ventilateur": 800,
    "Pompe à perfusion": 600,
    "Scanner": 1500,
    "IRM": 2000,
    "Autoclave": 1200,
    "Moniteur patient": 1000,
    "Rayons X": 1200,
    "Incubateur néonatal": 1000,
    "Machine de dialyse": 1000,
    "Bistouri électrique": 1000,
    "Aspirateur chirurgical": 500,
    "Échographe": 1200,
    "Pousse-seringue": 600
}

rows = [
    # --- Critique (12) ---
    ("Défibrillateur", "Critique",
     "Défibrillateur en panne : impossible de délivrer le choc malgré batterie pleine. Alarme critique affichée.",
     "Defibrillator failure: unable to deliver shock despite full battery. Critical alarm shown."),
    ("Ventilateur", "Critique",
     "Ventilateur de réanimation : fuite d’oxygène détectée, odeur de brûlé, appareil arrêté d'urgence.",
     "ICU ventilator: oxygen leak detected, burning smell, device stopped in emergency."),
    ("Pompe à perfusion", "Critique",
     "Pompe à perfusion délivre une dose erronée (+30%). Alarme 'dose excessive'.",
     "Infusion pump delivers wrong dose (+30%). 'Excess dose' alarm."),
    ("Scanner", "Critique",
     "Scanner : surchauffe de l’alimentation, fumée légère, arrêt immédiat requis.",
     "CT scanner: power supply overheating, light smoke, immediate shutdown required."),
    ("IRM", "Critique",
     "IRM : bruit anormal du compresseur cryogénique, température de l’aimant hors seuil.",
     "MRI: abnormal cryo compressor noise, magnet temperature out of range."),
    ("Autoclave", "Critique",
     "Autoclave : cycle de stérilisation échoué, capteur de pression défectueux.",
     "Autoclave: sterilization cycle failed, faulty pressure sensor."),
    ("Moniteur patient", "Critique",
     "Moniteur patient : arrêts aléatoires en per-opératoire, écran noir.",
     "Patient monitor: random shutdowns during surgery, black screen."),
    ("Rayons X", "Critique",
     "Générateur RX : tension du tube hors tolérance, risque de surexposition.",
     "X-ray generator: tube voltage out of tolerance, risk of overexposure."),
    ("Incubateur néonatal", "Critique",
     "Incubateur : température dépasse 39°C, alarme haute persistante.",
     "Neonatal incubator: temperature exceeds 39°C, persistent high alarm."),
    ("Machine de dialyse", "Critique",
     "Dialyse : fuite de dialysat, conductivité hors plage.",
     "Dialysis machine: dialysate leak, conductivity out of range."),
    ("Bistouri électrique", "Critique",
     "Bistouri électrique : défaut d’isolation détecté, risque de brûlure patient.",
     "Electrosurgical unit: insulation fault detected, risk of patient burn."),
    ("Aspirateur chirurgical", "Critique",
     "Aspirateur chirurgical : perte totale de vacuum pendant intervention.",
     "Surgical suction: total vacuum loss during procedure."),

    # --- Modérée (12) ---
    ("Défibrillateur", "Modérée",
     "Auto-test hebdomadaire : électrodes à remplacer bientôt, performances actuelles OK.",
     "Weekly self-test: pads to be replaced soon, current performance OK."),
    ("Ventilateur", "Modérée",
     "Filtre expiratoire proche de fin de vie, pressions stables.",
     "Expiratory filter near end-of-life, pressures stable."),
    ("Pompe à perfusion", "Modérée",
     "Batterie se décharge rapidement; utilisation sur secteur recommandée.",
     "Battery drains quickly; recommend mains operation."),
    ("Scanner", "Modérée",
     "Calibration mensuelle due, image légèrement bruitée.",
     "Monthly calibration due, image slightly noisy."),
    ("IRM", "Modérée",
     "Message intermittent 'RF coil check' ; examen possible après redémarrage.",
     "Intermittent 'RF coil check' message; exam possible after reboot."),
    ("Autoclave", "Modérée",
     "Joint de porte usé, petite fuite vapeur en fin de cycle, stérilisation conforme.",
     "Door gasket worn, small steam leak end of cycle, sterilization compliant."),
    ("Moniteur patient", "Modérée",
     "Capteur SpO2 instable chez certains patients; câble probablement usé.",
     "SpO2 sensor unstable for some patients; cable likely worn."),
    ("Rayons X", "Modérée",
     "Bouton d'exposition dur; lubrification ou remplacement recommandé.",
     "Exposure button stiff; lubrication or replacement recommended."),
    ("Incubateur néonatal", "Modérée",
     "Ventilateur interne bruyant par moments; températures dans la plage.",
     "Internal fan occasionally noisy; temperatures within range."),
    ("Machine de dialyse", "Modérée",
     "Alerte maintenance préventive prévue dans 72 h.",
     "Preventive maintenance due in 72 h."),
    ("Échographe", "Modérée",
     "Trackball dur; dérive doppler minime observée.",
     "Trackball stiff; minor doppler drift observed."),
    ("Pousse-seringue", "Modérée",
     "Firmware obsolète; mise à jour recommandée.",
     "Outdated firmware; update recommended."),

    # --- Faible (12) ---
    ("Défibrillateur", "Faible",
     "Cache batterie fendu (esthétique). Auto-test OK.",
     "Battery cover cracked (cosmetic). Self-test OK."),
    ("Ventilateur", "Faible",
     "Étiquette inventaire illisible; à réimprimer.",
     "Inventory label unreadable; needs reprint."),
    ("Pompe à perfusion", "Faible",
     "Support de fixation lâche; resserré par l’équipe, fonctionnement normal.",
     "Mounting bracket loose; tightened by staff, normal operation."),
    ("Scanner", "Faible",
     "Demande de nettoyage clavier/console et filtres d'air.",
     "Request to clean keyboard/console and air filters."),
    ("IRM", "Faible",
     "Coussins de positionnement manquants; commande consommables en cours.",
     "Positioning pads missing; consumables order in progress."),
    ("Autoclave", "Faible",
     "Bruit léger de porte; aucun impact sur le cycle.",
     "Slight door noise; no impact on cycle."),
    ("Moniteur patient", "Faible",
     "Roulettes du pied à perfusion grincent; lubrification demandée.",
     "IV pole wheels squeak; lubrication requested."),
    ("Rayons X", "Faible",
     "Capot plastique rayé; fonctionnement nominal.",
     "Plastic cover scratched; normal operation."),
    ("Incubateur néonatal", "Faible",
     "Éclairage de veille défectueux; non utilisé cliniquement.",
     "Standby light faulty; not clinically used."),
    ("Machine de dialyse", "Faible",
     "Création d’un nouveau profil patient demandée; pas de panne.",
     "New patient profile creation requested; no failure."),
    ("Échographe", "Faible",
     "Formation utilisateur demandée sur la mesure B; appareil OK.",
     "User training requested on B-mode; device OK."),
    ("Pousse-seringue", "Faible",
     "Câble secteur manquant; prêt d’un câble disponible.",
     "Mains cable missing; loaner cable available.")
]

# Build DataFrame
rng = np.random.RandomState(42)
types = []
alertes = []
fr_texts = []
en_texts = []
tf_vals = []
freq_vals = []

for t, a, fr, en in rows:
    types.append(t)
    alertes.append(a)
    fr_texts.append(fr)
    en_texts.append(en)
    # Randomize hours of operation; ensure frequency from mapping
    freq = freq_by_type.get(t, 1000)
    freq_vals.append(freq)
    tf_vals.append(int(rng.randint(low=80, high=6000)))

df = pd.DataFrame({
    "Type": types,
    "Alerte": alertes,
    "Rapport_FR": fr_texts,
    "Report_EN": en_texts,
    "Temps de fonctionnement": tf_vals,
    "Fréquence maintenance": freq_vals
})

# Save and display
path = "C:/HealthPredict AI/data/healthpredict_rapports_test_fr_en.xlsx"
df.to_csv(path, index=False, encoding="utf-8")

# Try to display nicely
try:
    from ace_tools import display_dataframe_to_user
    display_dataframe_to_user("Exemples FR/EN pour tester la prédiction (aperçu)", df.head(12))
except Exception as e:
    # Fallback plain print of head
    df.head(12)
    print(df.head(12))
    print(f"Error displaying DataFrame: {e}")   
print(f"Test dataset saved to {path}")
print("Dataset creation complete.") 
