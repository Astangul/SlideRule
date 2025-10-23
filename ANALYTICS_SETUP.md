# Guide : Mise en place de Google Analytics pour SlideRule

## Étapes pour activer Google Analytics

### 1. Créer un compte Google Analytics (si vous n'en avez pas)

1. Allez sur https://analytics.google.com/
2. Connectez-vous avec votre compte Google
3. Cliquez sur "Commencer la mesure"
4. Créez un compte Analytics :
   - Nom du compte : "SlideRule"
   - Cochez les options de partage de données selon vos préférences
5. Créez une propriété :
   - Nom de la propriété : "SlideRule App"
   - Fuseau horaire : Europe/Paris
   - Devise : Euro
6. Configurez un flux de données Web :
   - URL du site Web : https://sliderule.streamlit.app/
   - Nom du flux : "SlideRule Production"
7. **Récupérez votre ID de mesure** (format : G-XXXXXXXXXX)

### 2. Ajouter Google Analytics à votre application

#### Option A : Via un fichier de configuration (Recommandé)

Créez un fichier `secrets.toml` dans le dossier `.streamlit/` :

```toml
# .streamlit/secrets.toml
[analytics]
google_analytics_id = "G-XXXXXXXXXX"  # Remplacez par votre vrai ID
```

**Important** : Ajoutez ce fichier à `.gitignore` pour ne pas exposer votre ID publiquement !

Puis dans `00_👋_SlideRule_app.py`, ajoutez après les imports :

```python
from utils.analytics import inject_google_analytics

# Injecter Google Analytics (si l'ID est configuré)
try:
    ga_id = st.secrets["analytics"]["google_analytics_id"]
    inject_google_analytics(ga_id)
except Exception:
    pass  # Pas d'analytics en développement local
```

#### Option B : Via Streamlit Community Cloud (Plus sécurisé)

1. Allez sur https://share.streamlit.io/
2. Cliquez sur votre application SlideRule
3. Cliquez sur "⚙️ Settings"
4. Dans la section "Secrets", ajoutez :
   ```toml
   [analytics]
   google_analytics_id = "G-XXXXXXXXXX"
   ```
5. Cliquez sur "Save"

### 3. Tracker des événements personnalisés (optionnel)

Vous pouvez tracker des actions spécifiques, par exemple :

```python
from utils.analytics import track_custom_event

# Quand un utilisateur charge des données
if selected_data == "Bare configurations":
    track_custom_event(
        event_name="data_selection",
        event_category="configuration",
        event_label="bare"
    )

# Quand un utilisateur génère un graphique
track_custom_event(
    event_name="plot_generated",
    event_category="visualization",
    event_label="scatter_plot"
)
```

### 4. Voir vos statistiques

1. Allez sur https://analytics.google.com/
2. Sélectionnez votre propriété "SlideRule App"
3. Vous verrez :
   - **Temps réel** : Visiteurs actuellement sur le site
   - **Acquisition** : D'où viennent vos visiteurs
   - **Engagement** : Pages visitées, durée des sessions
   - **Données démographiques** : Pays, villes, langues
   - **Technologie** : Navigateurs, appareils, OS

### 5. Rapports utiles

#### Nombre de visiteurs uniques par jour
- Allez dans "Rapports" > "Engagement" > "Pages et écrans"
- Ajustez la période en haut à droite

#### Localisation géographique
- Allez dans "Rapports" > "Utilisateur" > "Données démographiques" > "Détails démographiques"
- Sélectionnez "Pays" ou "Ville"

#### Pages les plus visitées
- Allez dans "Rapports" > "Engagement" > "Pages et écrans"
- Triez par "Vues"

## Alternatives à Google Analytics

### 1. **Plausible Analytics** (Payant mais respectueux de la vie privée)
- Plus simple et axé sur la confidentialité
- Pas de cookies, conforme RGPD
- ~9€/mois pour 10k vues/mois
- https://plausible.io/

### 2. **Simple Analytics** (Payant)
- Très simple, pas de cookies
- ~19€/mois
- https://simpleanalytics.com/

### 3. **Umami** (Gratuit et open-source)
- Auto-hébergé (nécessite un serveur)
- Respectueux de la vie privée
- https://umami.is/

### 4. **Tracking personnalisé avec Streamlit Session State**

Si vous voulez juste des statistiques basiques sans service externe :

```python
# Dans chaque page, à la fin
import datetime
import json
from pathlib import Path

def log_visit(page_name):
    """Enregistre une visite dans un fichier JSON local"""
    log_file = Path("logs/visits.json")
    log_file.parent.mkdir(exist_ok=True)
    
    visit_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "page": page_name,
        "session_id": st.session_state.get("session_id", "unknown")
    }
    
    # Lire les logs existants
    logs = []
    if log_file.exists():
        with open(log_file, "r") as f:
            logs = json.load(f)
    
    # Ajouter le nouveau log
    logs.append(visit_data)
    
    # Sauvegarder
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=2)

# Générer un ID de session unique
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# Logger la visite
log_visit("Raw_results")
```

**Note** : Cette méthode ne fonctionne PAS sur Streamlit Community Cloud car le système de fichiers est éphémère.

## Recommandation

Pour votre cas (application publique sur Streamlit Community Cloud), je recommande **Google Analytics** car :

✅ Gratuit  
✅ Complet (localisation, statistiques détaillées)  
✅ Facile à intégrer  
✅ Données en temps réel  
✅ Exportation des données possible  
✅ Compatible avec Streamlit Community Cloud  

Voulez-vous que je vous aide à l'implémenter dans votre application ?
