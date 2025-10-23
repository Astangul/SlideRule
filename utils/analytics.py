# utils/analytics.py
import streamlit as st
import streamlit.components.v1 as components

def inject_google_analytics(ga_id):
    """
    Injecte Google Analytics dans l'application Streamlit.
    
    Args:
        ga_id (str): Votre ID Google Analytics (format: G-XXXXXXXXXX)
    """
    
    # Code Google Analytics avec propagation au document parent
    ga_html = f"""
    <script>
        // Créer le script gtag.js dans le document parent (page principale)
        (function() {{
            // Vérifier si déjà chargé
            if (window.parent.gtag) {{
                return;
            }}
            
            // Créer et injecter le script gtag.js dans le parent
            var script = window.parent.document.createElement('script');
            script.async = true;
            script.src = 'https://www.googletagmanager.com/gtag/js?id={ga_id}';
            window.parent.document.head.appendChild(script);
            
            // Initialiser dataLayer dans le parent
            window.parent.dataLayer = window.parent.dataLayer || [];
            function gtag(){{window.parent.dataLayer.push(arguments);}}
            window.parent.gtag = gtag;
            
            gtag('js', new Date());
            gtag('config', '{ga_id}');
            
            console.log('Google Analytics chargé: {ga_id}');
        }})();
    </script>
    """
    
    # Injecter dans un iframe invisible qui modifie le parent
    components.html(ga_html, height=0)


def track_page_view(page_name):
    """
    Enregistre une vue de page personnalisée.
    
    Args:
        page_name (str): Nom de la page visitée
    """
    # Le tracking est automatique avec Google Analytics
    # Cette fonction peut être utilisée pour des événements personnalisés
    pass


def track_custom_event(event_name, event_category, event_label=None, value=None):
    """
    Enregistre un événement personnalisé.
    
    Args:
        event_name (str): Nom de l'événement
        event_category (str): Catégorie de l'événement
        event_label (str, optional): Label de l'événement
        value (int, optional): Valeur numérique associée
    """
    
    event_params = {
        'event_category': event_category,
    }
    
    if event_label:
        event_params['event_label'] = event_label
    if value is not None:
        event_params['value'] = value
    
    js_code = f"""
    <script>
        gtag('event', '{event_name}', {{
            'event_category': '{event_category}',
            {'event_label': '" + event_label + "', ' if event_label else ''}
            {'value': " + str(value) if value is not None else ''}
        }});
    </script>
    """
    
    components.html(js_code, height=0)
