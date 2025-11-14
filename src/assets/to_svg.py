import cv2 as cv2
import numpy as np

def contours_to_svg(contours, filename="out.svg", width=None, height=None):
    # Breite/Höhe optional für viewBox
    if width is None or height is None:
        width = max([p[0][0] for c in contours for p in c]) + 1
        height = max([p[0][1] for c in contours for p in c]) + 1

    with open(filename, "w") as f:
        f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n')
        for cnt in contours:
            # "M x y L x y ..." Path
            d = "M " + " L ".join(f"{p[0][0]} {p[0][1]}" for p in cnt) + " Z"
            f.write(f'<path d="{d}" stroke="black" fill="none"/>\n')
        f.write('</svg>')
        


img = cv2.imread('C:\\Users\\muham\\repos\\DrawMeMaybe\\src\\assets\\BCO.png', cv2.IMREAD_GRAYSCALE)

#pixel in liste umwandeln
pixels = img.reshape((-1, 1)).astype(np.float32)

#Kmeans parameter
# Define criteria = ( type(beide ausgewählt), max_iter = 30 , epsilon = 0.2 (min änderung nach iteration) )
criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.2)
_k, labels, centers = cv2.kmeans(
    pixels,           # Daten
    3,                # Anzahl Cluster (hell, mittel, dunkel)
    None,             # Keine Initiallabels
    criteria,         # Kriterien
    10,               # Anzahl der Initialisierungen (versuche)
    cv2.KMEANS_RANDOM_CENTERS
)
#cluster zurück ins bild
segmented = centers[labels.flatten()].reshape(img.shape).astype("uint8")

blurred = cv2.GaussianBlur(segmented, (5, 5), 0) #canny sonst empfindlich gegen Bildrauschen und krassen Kontrast
edges = cv2.Canny(blurred, 80, 150)  #thresholds1 = vllt Kante, threshold2 = starke Kante


#konturen finden
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
top_n = 150
top_contours = contours[:top_n]

#richtung SVG
approx_contours = [cv2.approxPolyDP(cnt, epsilon=2, closed=True) for cnt in top_contours]
contours_to_svg(approx_contours, "top150.svg")


canvas = np.ones_like(edges) * 255  # weißer Hintergrund
cv2.drawContours(canvas, top_contours, -1, (0,0,0), 1)  # schwarze Linien, dicke=1

cv2.imshow("Top 100 Contours", cv2.resize(canvas,None, fx=0.8, fy=0.8))
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.imshow("Original", cv2.resize(img,None, fx=0.4, fy=0.4))
cv2.imshow("Segmented", cv2.resize(segmented,None, fx=0.4, fy=0.4))
cv2.imshow("blurred", cv2.resize(blurred,None, fx=0.4, fy=0.4))
cv2.imshow("edges", cv2.resize(edges,None, fx=0.8, fy=0.8))
cv2.waitKey(0)
cv2.destroyAllWindows()




# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
# gray = cv2.GaussianBlur(gray, (5, 5), 0) #canny sonst empfindlich gegen Bildrauschen und krassen Kontrast

# edges = cv2.Canny(gray, 80, 150)  #thresholds1 = vllt Kante, threshold2 = starke Kante
# edges2 = cv2.dilate(edges, None)  #dilation macht helle pixel größer, verbindet kanten
# edges3 = cv2.erode(edges2, None)

# cv2.imshow("edges", cv2.resize(edges,None, fx=0.2, fy=0.2))
# cv2.imshow("edges2", cv2.resize(edges2,None, fx=0.2, fy=0.2)) 
# cv2.imshow("edges3", cv2.resize(edges3,None, fx=0.2, fy=0.2))
# cv2.waitKey(0)
# cv2.destroyAllWindows()



