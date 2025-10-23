# 🚀 Guide de Configuration Google Analytics pour SlideRule

## ✅ Ce qui est déjà fait

- ✅ Code Google Analytics intégré dans `00_👋_SlideRule_app.py`
- ✅ Fichier `utils/analytics.py` créé avec les fonctions de tracking
- ✅ Fichier `.streamlit/secrets.toml` créé (protégé par `.gitignore`)

## 📋 Étapes à suivre

### ÉTAPE 1 : Créer votre compte Google Analytics (5 minutes)

1. **Allez sur** : https://analytics.google.com/
2. **Connectez-vous** avec votre compte Google (personnel ou professionnel)
3. **Cliquez sur** "Commencer la mesure" ou "Créer un compte"

### ÉTAPE 2 : Configurer votre compte Analytics

**Configuration du compte** :
- Nom du compte : `SlideRule` (ou le nom de votre choix)
- Cochez les options de partage de données selon vos préférences
- Cliquez sur "Suivant"

**Configuration de la propriété** :
- Nom de la propriété : `SlideRule App`
- Fuseau horaire : `(GMT+01:00) Europe/Paris`
- Devise : `Euro (EUR)`
- Cliquez sur "Suivant"

**À propos de votre entreprise** (optionnel) :
- Sélectionnez vos options (ou passez)
- Cliquez sur "Créer"

**Acceptez les conditions d'utilisation**

### ÉTAPE 3 : Créer un flux de données Web

1. **Sélectionnez** "Web" comme plateforme
2. **Configurez le flux de données** :
   - **URL du site web** : `https://sliderule.streamlit.app`
   - **Nom du flux** : `SlideRule Production`
   - Cochez "Activer la mesure améliorée" (recommandé)
3. **Cliquez sur** "Créer un flux"

### ÉTAPE 4 : Récupérer votre ID de mesure

Après la création du flux, vous verrez votre **ID de mesure** au format :

```
G-XXXXXXXXXX
```

**Exemple** : `G-ABC123DEF4`

⚠️ **IMPORTANT** : Copiez cet ID, vous en aurez besoin pour les étapes suivantes !

### ÉTAPE 5 : Configurer l'ID dans votre application locale

Ouvrez le fichier `.streamlit/secrets.toml` et remplacez :

```toml
[analytics]
google_analytics_id = "VOTRE_ID_ICI"
```

Par :

```toml
[analytics]
google_analytics_id = "G-ABC123DEF4"  # ⬅️ Votre vrai ID ici
```

💾 **Sauvegardez** le fichier.

### ÉTAPE 6 : Configurer l'ID sur Streamlit Community Cloud

1. **Allez sur** : https://share.streamlit.io/
2. **Connectez-vous** avec votre compte GitHub
3. **Cliquez sur** votre application `SlideRule`
4. **Cliquez sur** l'icône "⚙️" (Settings)
5. **Dans la section "Secrets"**, ajoutez :

```toml
[analytics]
google_analytics_id = "G-ABC123DEF4"
```

⚠️ **Remplacez** `G-ABC123DEF4` par votre vrai ID !

6. **Cliquez sur** "Save"
7. L'application va redémarrer automatiquement

### ÉTAPE 7 : Vérifier que ça fonctionne

#### Test en local :
1. Lancez votre application localement : `streamlit run 00_👋_SlideRule_app.py`
2. Ouvrez votre navigateur sur `http://localhost:8501`
3. Allez sur Google Analytics → Rapports → Temps réel
4. Vous devriez voir 1 utilisateur actif (vous !)

#### Test sur Streamlit Cloud :
1. Allez sur `https://sliderule.streamlit.app/`
2. Consultez Google Analytics → Temps réel
3. Vous devriez voir votre visite

## 📊 Voir vos statistiques

### Rapports en temps réel
- **Où** : Google Analytics → Rapports → Temps réel
- **Info** : Visiteurs actuellement sur le site

### Visiteurs par jour
- **Où** : Rapports → Acquisition → Vue d'ensemble de l'acquisition
- **Info** : Nombre de visiteurs quotidiens, hebdomadaires, mensuels

### Localisation géographique
- **Où** : Rapports → Utilisateur → Données démographiques → Détails démographiques
- **Info** : Pays, villes d'où viennent vos visiteurs

### Pages les plus visitées
- **Où** : Rapports → Engagement → Pages et écrans
- **Info** : Quelles pages de votre app sont les plus utilisées

### Données techniques
- **Où** : Rapports → Tech → Vue d'ensemble de la technologie
- **Info** : Navigateurs, systèmes d'exploitation, appareils utilisés

## 🎯 Événements personnalisés (optionnel)

Si vous voulez tracker des actions spécifiques (par exemple : "Utilisateur a sélectionné Bare configurations"), vous pouvez ajouter dans vos pages :

```python
from utils.analytics import track_custom_event

# Exemple : tracker la sélection d'une configuration
track_custom_event(
    event_name="configuration_selected",
    event_category="user_interaction",
    event_label="bare_configurations"
)
```

Ces événements apparaîtront dans : **Rapports → Engagement → Événements**

## ❓ Problèmes courants

### Je ne vois aucune donnée dans Google Analytics

**Vérifications** :
1. L'ID de mesure est-il correct dans `.streamlit/secrets.toml` ?
2. Avez-vous redémarré l'application après avoir configuré les secrets ?
3. Attendez 24-48h pour les premiers rapports complets (le temps réel fonctionne immédiatement)

### Les statistiques ne s'affichent pas sur Streamlit Cloud

**Vérifications** :
1. Avez-vous bien ajouté les secrets sur https://share.streamlit.io/ ?
2. L'application a-t-elle bien redémarré après l'ajout des secrets ?
3. Vérifiez les logs de l'application pour voir s'il y a des erreurs

### Je veux désactiver temporairement Google Analytics

Dans `.streamlit/secrets.toml`, commentez ou supprimez la ligne :

```toml
[analytics]
# google_analytics_id = "G-ABC123DEF4"  # ⬅️ Commenté = désactivé
```

## 🔒 Confidentialité et RGPD

Google Analytics collecte :
- Pages visitées
- Durée des sessions
- Localisation approximative (pays, ville)
- Type de navigateur et appareil
- Langue du navigateur

**Il ne collecte PAS** :
- Adresses email
- Noms d'utilisateurs
- Données personnelles saisies dans l'application

Si votre application est utilisée dans l'UE, vous devriez :
- Ajouter une mention "Cookies" dans votre page d'accueil
- Informer les utilisateurs que Google Analytics est utilisé
- Proposer un moyen de refuser le tracking

## 📞 Besoin d'aide ?

- Documentation Google Analytics : https://support.google.com/analytics
- Tutoriel vidéo (FR) : https://www.youtube.com/watch?v=kc3TsLQWq_Y
- Forum Streamlit : https://discuss.streamlit.io/

---

Bonne chance ! 🚀
