# Music Splitter — Contexte du projet

## Vision
Application web 100% client-side (hébergeable sur GitHub Pages) qui permet de séparer les instruments/voix d'un fichier audio (MP3, FLAC, WAV...) puis de mixer les stems en temps réel.

## Décisions prises
- **Stack** : Web app statique (HTML/CSS/JS vanilla), pas de framework, pas de backend
- **Déploiement** : GitHub Pages (tout tourne dans le navigateur)
- **Modèle ML** : **Demucs v4 (htdemucs)** via ONNX Runtime Web — 4 stems
- **Bibliothèque** : `demucs-web` (timcsy, MIT) — code ES6 vanilla embarqué dans `js/demucs/`
- **Mode temps réel** : Lecture interactive (charger → séparer → mixer avec mute/solo/volume par piste)
- **Stems** : 4 — voix, basse, batterie, autre (instruments mélangés)
- **Machine de Constantin** : Mac Apple Silicon (principal) + PC Windows à côté

## État actuel (v2)
L'app utilise une vraie séparation ML via Demucs v4.

```
Music Splitter/
├── index.html              # SPA — 3 vues : upload → processing → player
├── coi-serviceworker.js    # Active COOP/COEP pour SharedArrayBuffer (ONNX multi-thread sur GH Pages)
├── css/style.css           # Dark theme style production musicale
├── js/
│   ├── app.js              # Contrôleur principal (ES module), orchestre Demucs + UI
│   ├── audio-engine.js     # Web Audio API : lecture sync (44.1 kHz), mix, export WAV
│   ├── waveform.js         # Rendu Canvas des formes d'onde
│   └── demucs/             # Port ES6 de demucs-web (MIT, timcsy)
│       ├── index.js        # Exports publics
│       ├── constants.js    # SAMPLE_RATE, FFT_SIZE, model URL, etc.
│       ├── fft.js          # STFT/iSTFT Cooley-Tukey radix-2
│       └── processor.js    # DemucsProcessor : load ONNX, segment, infer, overlap-add
```

### Ce qui marche
- Upload drag & drop (MP3, FLAC, WAV, OGG, M4A, max 50 Mo)
- **Séparation ML réelle** : modèle Demucs v4 htdemucs (~172 Mo, téléchargé depuis Hugging Face au premier usage, ensuite caché par le navigateur)
- Backend : WebGPU si dispo (Chrome/Edge récents, Safari récent), sinon WASM multi-thread
- Lecteur interactif : mute/solo/volume par stem, waveforms animées, playhead
- Export WAV : stems individuels, mix custom selon volumes/mutes, ou tous en batch
- Raccourcis clavier : espace (play/pause), flèches (±5s), L (loop)
- Mode alternatif : charger des stems déjà séparés (fichiers multiples)
- **Fallback gracieux** : si le modèle ne se charge pas (offline, pas de SharedArrayBuffer), repli sur l'ancienne séparation par filtres fréquentiels avec avertissement
- **Toggle "sous-diviser Autre"** : case à cocher sur la page upload. Si activée, le stem "Autre" est post-traité après Demucs en 3 sous-bandes (grave <800Hz, médium 800-3kHz, aigu >3kHz) via filtres biquad. Utile pour chiptune, synthwave, OST jeux vidéo rétro où "Autre" est un fourre-tout. La séparation reste imparfaite (filtres EQ, pas IA) mais résout les cas où piano/lead/pads jouent dans des registres distincts. 4 stems deviennent 6 (vocals, drums, bass, other_low, other_mid, other_high).

### Ce qui reste à faire (v3+)
1. **Pré-séparation guitare/piano** — Le htdemucs standard fait 4 stems. Pour 6 stems (ajout guitare + piano), il faudrait htdemucs_6s en ONNX (pas encore packagé proprement à notre connaissance ; à investiguer le projet GSoC 2025 Mixxx).
2. **Web Worker** — Actuellement la séparation tourne sur le main thread. Le déplacer dans un worker libérerait l'UI pendant l'inférence (3-5 min sur CPU).
3. **Améliorer l'UI/UX** — Responsive mobile, animations, thèmes
4. **Format d'export MP3** — Actuellement WAV uniquement (MP3 nécessiterait lamejs ou similaire)
5. **Sauvegarde de presets de mix**
6. **Cache modèle via IndexedDB** — Pour ne pas dépendre du cache HTTP du navigateur.

## Notes techniques
- **COOP/COEP** : ONNX Runtime Web a besoin de `SharedArrayBuffer` (multi-thread WASM). Pour que ça fonctionne sur GitHub Pages sans contrôler les en-têtes serveur, `coi-serviceworker.js` (gzuidhof, MIT) s'enregistre et intercepte les fetch pour ajouter `Cross-Origin-Opener-Policy: same-origin` et `Cross-Origin-Embedder-Policy: require-corp`. La première visite déclenche un reload automatique pour que le SW prenne le contrôle.
- **Module ES6** : `app.js` est chargé en `type="module"` ; il importe ONNX Runtime depuis `cdn.jsdelivr.net` et `DemucsProcessor` depuis `./demucs/index.js`. Les fichiers `waveform.js` et `audio-engine.js` restent en scripts classiques (classes globales).
- **Sample rate** : tout le pipeline tourne en 44.1 kHz (sample rate natif de Demucs). L'AudioContext de l'`AudioEngine` est forcé à 44100. Si le fichier source est à un autre taux, `decodeAudioData` resample automatiquement, mais on a aussi un fallback `resample()` linéaire dans `app.js` au cas où.
- **Segmentation** : Demucs traite l'audio par fenêtres de ~7.8 s (`TRAINING_SAMPLES / SAMPLE_RATE`) avec un overlap de 25%. L'overlap-add est weighted (fade-in/fade-out triangulaire) pour éviter les artefacts aux frontières.
- **Modèle hébergé** : `https://huggingface.co/timcsy/demucs-web-onnx/resolve/main/htdemucs_embedded.onnx` (172 Mo). Hugging Face renvoie le bon `Content-Length` donc le téléchargement affiche une progression réelle.
- **Performance** : sur Mac Apple Silicon avec WebGPU, compter ~30s-1min pour un morceau de 3-4 min. Sur WASM (CPU), 3-5 min.
