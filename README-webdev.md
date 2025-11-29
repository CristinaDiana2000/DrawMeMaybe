
# Für Camera auf Hady/Tablet

## Android + Chrome: Unsichere Origin als sicher behandeln

Auf dem Handy (Android + Chrome): Origin „unsicher, aber als sicher behandeln“

Auf deinem Android-Handy in Chrome:

In die Adresszeile eingeben:
chrome://flags/#unsafely-treat-insecure-origin-as-secure

Es öffnet sich die „Experiments / Flags“ Seite, Flag ist bereits markiert.

Bei dem Flag „Unsafely treat insecure origin as secure“:

auf Enabled stellen.

Im Feld darunter (oder in einem Dialog) trägst du deine Dev-URL ein, z.B.:

<http://192.168.0.15:5173>

Oben auf Relaunch / „Browser neu starten“ tippen.

Dann:

Wieder Chrome öffnen,

<http://192.168.0.15:5173> aufrufen

Deine App laden → die Seite wird jetzt wie HTTPS behandelt

Kamera-Zugriff sollte erlaubt werden → navigator.mediaDevices.getUserMedia ist da, und in deiner CameraView sollte die Browser-Kameraabfrage kommen.
