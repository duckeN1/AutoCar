import numpy as np
import math
import itertools as it


while(1):

    tocke = int(input("Vnesite željeno število točk: "))
    if tocke == 3:
        print("Prva točka")
        p1x = int(input("X prve točke: "))
        p1y = int(input("Y prve točke: "))
        p1z = int(input("Z prve točke: "))

        print("Druga točka")
        p2x = int(input("X druge točke: "))
        p2y = int(input("Y druge točke: "))
        p2z = int(input("Z druge točke: "))

        print("Tretja točka")
        p3x = int(input("X tretje točke: "))
        p3y = int(input("Y tretje točke: "))
        p3z = int(input("Z tretje točke: "))
        

        koordinate1 = np.array([p1x, p1y, p1z])
        koordinate2 = np.array([p2x, p2y, p2z])
        koordinate3 = np.array([p3x, p3y, p3z])

        print("Razdalje: ")
        r1 = float(input("Prva razdalja: "))
        r2 = float(input("Druga razdalja: "))
        r3 = float(input("Tretja razdalja: "))

        P1 = np.subtract(koordinate1, koordinate1)
        P2 = np.subtract(koordinate2, koordinate1)
        P3 = np.subtract(koordinate3, koordinate1)

        V1 = np.subtract(P2, P1)
        V2 = np.subtract(P3, P1)
        Xn = V1 / np.linalg.norm(V1)

        

        Zn = np.cross(V1, V2)#cross je vektorski produkt .. arrayev
        #(V1 x V2) / |V1 x V2|
        Zn = Zn / np.linalg.norm(Zn)

        Yn = np.cross(Xn, Zn)
        #Zn x Xn
        d = np.dot(Xn, V1)
        i = np.dot(Xn, V2)
        j = np.dot(Yn, V2)
        #kjer je d = Xn · V1 = |V1|, i = Xn · V2 in j = Yn · V2

        X = (pow(r1, 2) - pow(r2, 2) + pow(d, 2)) / (2 * d)
        Y = (pow(r1, 2) - pow(r3, 2) + pow(i, 2) + pow(j, 2)) / (2 * j) - (i / j) * X
        Z = math.sqrt(abs(pow(r1, 2) - pow(X, 2) - pow(Y, 2)))

        K1 = koordinate1 + (X * Xn) + (Y * Yn) + (Z * Zn)
        K2 = koordinate1 + (X * Xn) + (Y * Yn) - (Z * Zn)


        print("Dvoumen položaj:  ", K1)
        print("Dvoumen položaj:  ", K2)
    elif tocke == 4:
        print("Prva točka")
        p1x = int(input("X prve točke: "))
        p1y = int(input("Y prve točke: "))
        p1z = int(input("Z prve točke: "))

        print("Druga točka")
        p2x = int(input("X druge točke: "))
        p2y = int(input("Y druge točke: "))
        p2z = int(input("Z druge točke: "))

        print("Tretja točka")
        p3x = int(input("X tretje točke: "))
        p3y = int(input("Y tretje točke: "))
        p3z = int(input("Z tretje točke: "))
        
        print("Četrta točka")
        p4x = int(input("X četrte točke: "))
        p4y = int(input("Y četrte točke: "))
        p4z = int(input("Z četrte točke: "))

        koordinate1 = np.array([p1x, p1y, p1z])
        koordinate2 = np.array([p2x, p2y, p2z])
        koordinate3 = np.array([p3x, p3y, p3z])
        koordinate4 = np.array([p4x, p4y, p4z])

        print("Razdalje: ")
        r1 = float(input("Prva razdalja: "))
        r2 = float(input("Druga razdalja: "))
        r3 = float(input("Tretja razdalja: "))
        r4 = float(input("Četrta razdalja: "))

        P1 = np.subtract(koordinate1, koordinate1)
        P2 = np.subtract(koordinate2, koordinate1)
        P3 = np.subtract(koordinate3, koordinate1)

        V1 = np.subtract(P2, P1)
        V2 = np.subtract(P3, P1)

        Xn = V1/np.linalg.norm(V1)

        Zn = np.cross(V1, V2)
        Zn = Zn / np.linalg.norm(Zn)

        Yn = np.cross(Xn, Zn)

        d = np.dot(Xn, V1)
        i = np.dot(Xn, V2)
        j = np.dot(Yn, V2)

        a = np.dot((np.subtract(koordinate4, koordinate1)), Xn) 
        b = np.dot((np.subtract(koordinate4, koordinate1)), Yn)
        c = np.dot((np.subtract(koordinate4, koordinate1)), Zn)
        #a = (P4 - P1) * Xn
        #b = (P4 - P1) * Yn
        #c = (P4 - P1) * Zn

        X = (pow(r1, 2) - pow(r2, 2) + pow(d, 2)) / (2 * d)
        Y = (pow(r1, 2) - pow(r3, 2) + pow(i, 2) + pow(j, 2)) / (2 * j) - (i / j) * X
        Z = (pow(r1, 2) - pow(r4, 2) + pow(a, 2) + pow(b, 2) + pow(c, 2)) / (2 * c) - (a / c) * X - (b / c) * Y

        K1 = koordinate1 + (X * Xn) + (Y * Yn) + (Z * Zn)

        print("Natančen položaj: ", K1)


    nadaljujem = int(input("Ali želite vnesti nove točke? ( 0 - NE | 1 - DA)"))
    if(nadaljujem):
        continue
    else:
        break