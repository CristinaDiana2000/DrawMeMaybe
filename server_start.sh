#!/bin/bash
echo "-----Apache2-Server-Verwaltung------"
echo "------------------------------------"
echo "(1) -> Apache2-Server starten"
echo "(2) -> Apache2-Server stoppen"
echo "(3) -> Apache2-Server neu starten"
echo "(4) -> Konfiguration neu laden"
echo "------------------------------------"
echo " "
echo "Welche Option?"
read option

case $option in
    1)
        sudo systemctl start apache2
        ;;
    2)
        sudo systemctl stop apache2
        ;;
    3)
        sudo systemctl restart apache2
        ;;
    4)
        sudo systemctl reload apache2
        ;;    
    *)
        echo "Dies Option gibt es nicht, sorry. Nur 1,2,3,4"
        ;;
esac