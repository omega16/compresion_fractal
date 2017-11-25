#!/usr/bin/env python3

import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib import gridspec


"""

24-Nov-2017

Metodo de compresion de imagenes usando fractales (PIFS)



Proyecto echo para la clase de Topologia II en ESFM-IPN 
Omega16 (nombre omitido por privacidad)
https://github.com/omega16/compresion_fractal/

Ver Readme en git para encontrar instrucciones sobre las librerias usadas

"""







def escalar_2x2(d):
    n1 = int(d.shape[0]/2)
    n2 = int(d.shape[1]/2)

    aux = np.zeros((n1,n2),dtype=np.uint8)

    for i in range(0,n1):
        for j in range(0,n2):
            aux[i,j] = int(np.sum(d[2*i:2*(i+1),2*j:2*(j+1)])/4.0)
            
            
    return aux


def escalar(d,m):
    """
    d = imagen
    
    m = int
    
    
    Dada una imagen y un tamaño minimo , reduce la imagen a la mitad del tamaño usando interpolacion simple (se puede usar cv2.resize para lograrlo mas rapidamente) hasta llegar a un tamaño por debajo del minimo
    
    """
    
    while m<d.shape[0]:
        d = escalar_2x2(d)

    return d





def comparar_dos(a,a2,d,r):
    """
    
    INPUT:
    
    a = float               (a = suma de las intencidades de d)
    
    a2 = float              (a2 = suma de los cuadrados de las intencidades de d)
    
    d = np.array            (imagen en escala de grices, dominio)
    
    r = np.array            (imagen en escala de grices, rango) 
    
    
    
    
    OUTPUT:
    
    
    
    R,s,o
    
    R = float               (Distancia inducida por la uclidea sobre matrices)
    
    s = float               (parametro de mejor aproximacion de r usando d, en la forma r aprox = sd +o)
    
    o = float               (parametro de aproximacion de r usando d, en la forma r aprox = sd +o)
    
    
    """
    
    a = a.astype(float)
    a2=a2.astype(float)
    d=d.astype(float)
    r=r.astype(float)
    b2 = np.sum(r**2)
    b = np.sum(r)
    ab = np.sum(d*r)
    n = float(r.size)
    if (a2 - (a**2)) == 0:
        s = 0
        o = 1/n * b
    
    else :
        s = ((n*ab) - (a*b))/((n*a2) -(a**2))    
        o = (1.0/n) * (b- (s*a))
        
    if s>=1:
        s=0.8
    
    return abs( 1/n * (b2 + (s*( (s*a2) - (2*ab) + (2*o*a) ) ) + (o* ( ( n*o )-( 2*b ) ) ) ) )**(0.5), s, o
    
    
    
    
    
    
def roi(lista,imagen):
    """
    Dada una lista de la foma  [x,y,w,h], donde x,y son coordenas de la esquina superior de un rectangulo en una imagen y w,h son las longitudes horizontal y vertical respectivamente del rectangulo, regresa el rectangulo sobre la imagen
    
    """
    return imagen[lista[0]:lista[0]+lista[2],lista[1]:lista[1]+lista[3]]




def comparar_listas(R,D,img,epsi=0.001,tama = 16):
    """
    
    dadas listas R y D de rectangulos sobre img, se comparan todos los elementos de D con cada elemento de R y a cada R[i] se le asigna el elemento de mayor afinidad D[aux[i]]  y se regresa junto con valores de la mejor transformacion affin que (s,o) que convierte a D[aux[i]] en R[i]
    
    vale[i]=1 es indicativo de que la eleccion de D[aux[i]] cumple los criterios minimos requeridos por el usuario para enlazar D[aux[i]] con R[i]
    
   
    """
    
    
    n = D.shape[0]
    m = R.shape[0] 
    aux = np.zeros(m,dtype=int)
    s = np.zeros(m)
    o = np.zeros(m)
    s2 = np.zeros(n)
    o2 = np.zeros(n)
    a = [np.sum(roi(lista,img)) for lista in D]
    a2 = [np.sum(roi(lista,img)**2) for lista in D]
    aux2 = np.zeros(n)
    
    vale = np.zeros(m)
    
    for i in range(0,m):
        for j in range(0,n):
            aux2[j],s2[j],o2[j]=comparar_dos( a[j],a2[j],escalar(roi(D[j],img),R[i][2]),roi(R[i],img) )
        aux[i] = np.argmin(aux2)
        s[i] = s2[aux[i]]
        o[i] = o2[aux[i]]
        
        if aux2[aux[i]] < epsi or R[i][2]*R[i][3] < tama:
            vale[i] = 1
        
        
    
    return aux,s,o,vale
    
    


def partir_4(roi):
    """
    
    Dado un rectangulo roi en la forma roi =[x,y,w,h] se parte en cuatro pedazos y se regresan los cuatro rectangulos
        
    """

    aux1 = int(roi[2]/2)
    aux2 = int(roi[3]/2)
    
    roi2=[[roi[0]+(i*aux1),roi[1]+(j*aux2),aux1,aux2] for i in [0,1] for j in [0,1]]
    return roi2







def comprimir(im,epsi=0.001,tama = 17,k=8,k2=4):
    
    """
    
    Guia del algoritmo de compression usando fractales
    
    """
    

    
    D=[[x,y,k,k] for x in range(0,im.shape[0],k ) for y in range(0,im.shape[1],k) ]
    
    D=np.array(D)
    print("Dominios creados con exito")

    R=[[x,y,k2,k2] for x in range(0,im.shape[0],k2) for y in range(0,im.shape[1],k2) ]
    
    R=np.array(R)
    
    print("Rangos creados con exito")
    print("Comenzando iteracion")
    aux,s,o,vale = comparar_listas(R,D,im,epsi,tama)
    
    s=s[vale==1]
    o=o[vale==1]
    itera=0
    guarda=[]
    guarda += [R[vale==1]]
    dom =[D[aux[vale==1]]]
    while vale[vale<1].shape[0]>0:
        print("Iteracion: ",itera)
        print("Resultados : \n Indices :  \n {0}  \n Parametros: \n {1} \n {2} \n Validacion: \n {3} \n".format(aux,s,o,vale))

        R = R[vale<1]
        R2=[]
        for i in range(0,R.shape[0]):
            R2 += partir_4(R[i])
                
               

        R = np.array(R2)
        aux,s1,o1,vale = comparar_listas(R,D,im,epsi,tama)
        
        guarda += [R[vale==1]]
        s1=s1[vale==1]
        o1=o1[vale==1]
        dom += [D[aux[vale==1]]]
        s = np.concatenate((s,s1))
        o = np.concatenate((o,o1))
        itera +=1
    dom = np.concatenate(dom)
    R=np.concatenate(guarda)
    return dom,s,o,R 
    
    
    
    
    
    
    
def mostrar_cuadricula(re1,re2,im):
    """
    
    Dibuja los rectangulos re1, re2 sobre im y los muestra en pantalla hasta que se pulsa una tecla
    
    """    
    
    im2 = im.copy()
    im3 = im.copy()
    cv2.namedWindow('R',cv2.WINDOW_NORMAL)
    cv2.namedWindow('D',cv2.WINDOW_NORMAL)
    cv2.rectangle(im2,(re1[0],re1[1]),(re1[0]+re1[2],re1[1]+re1[3]),120,1)
    cv2.rectangle(im3,(re2[0],re2[1]),(re2[0]+re2[2],re2[1]+re2[3]),120,1)
    
    cv2.imshow('R',im2)
    cv2.imshow('D',im3)
    cv2.waitKey(0)
    cv2.destroyWindow('R')
    cv2.destroyWindow('D')





def cuadricula(R,D,im):
    """
    
    Dibuja las cuadriculas (listas de rectangulos ) R y D sobre im
    
    """
    im2 = im.copy()
    im3 = im.copy()
    for i in R:
        cv2.rectangle(im2,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),120,1)
    
    for i in D:
        cv2.rectangle(im3,(i[0],i[1]),(i[0]+i[2],i[1]+i[3]),120,1)
    
    
    return im2, im3    

 
 
 
        


def paso(R,D,s,o,im):
    """
    
    Paso usual en la descompresion de una imagen fractal
    
    """
    sal = []
    
    for i in range(0,R.shape[0]):
            aux = escalar(roi(D[i],im),R[i][2])
            sal.append(np.uint8((aux*s[i])+o[i]))
    #sal=np.concatenate(sal)
    
    return sal
    
    
def une(nuevo,im,R):

    """
    
    Une los trozos de una imagen descomprimiendose
    
    """

    im2 = im.copy()
    for i in range(0,R.shape[0]):
        im2[R[i][0]:R[i][0]+R[i][2],R[i][1]:R[i][1]+R[i][3]] = nuevo[i]
        
    return np.uint8(im2)
    
    
    
    
    
    
    
    
def decomprimir(R,D,s,o,itera=3,im=0,largo=64,ancho=64, mirar=0):
    
    sale=[]
    
    if type(im)==type(0):
        im = np.random.random_integers(0,255,(largo,ancho))
        im = np.uint8(im)

    for i in range(0,itera):
        sal=paso(R,D,s,o,im)
        im = une(sal,im,R)
        sale.append(im.copy())
    
    
    return sale
        

    
    
    
    
    
    
    
    
    
    
    
        
def mostrar(im1,im2,nombre1='D',nombre2='R'):
    """
    
    forma rapida de ver dos imagenes usando opencv
    
    """
    
    cv2.namedWindow(nombre1,cv2.WINDOW_NORMAL)
    cv2.namedWindow(nombre2,cv2.WINDOW_NORMAL)
    
    cv2.imshow(nombre1,im1)
    cv2.imshow(nombre2,im2)
    
    cv2.waitKey(0)
    cv2.destroyWindow(nombre1)
    cv2.destroyWindow(nombre2)
    
    
    
def driver(nombre,itera=1,mira=0,nombre2=0):
    """
    
    Dada la direccion de una imagen en un formato admitido por opencv, de tamaño nxn (es decir cuadrada) tal que n se puede dividir por 8, se comprime la imagen usando PIFS, se guarda en los archivos s,o,D,R y se descomprime
    itera = numero de iteraciones usada en la descomprension
    mira =1 activa el ver imagen por imagen al terminar la descompresion
    nombre2 = imagen opcional que se usa para descomprimir la imagen original.
    
    
    """
    itera=int(itera)
    mira = int(mira)
    im=cv2.imread(nombre,0)
    print("comprimiendo")
    D,s,o,R = comprimir(im,epsi=0.6,tama = 17,k1=8,k2=4)
    print(D.shape,s.shape,o.shape,R.shape)
                   
    np.save(nombre+'_comprimido_'+str(itera)+'_s',s)
    np.save(nombre+'_comprimido_'+str(itera)+'_o',o)
    np.save(nombre+'_comprimido_'+str(itera)+'_D',D)
    np.save(nombre+'_comprimido_'+str(itera)+'_R',R)
    
    
    
    if type(nombre2)==type(0):
        dummy = 0
    else:
        dummy = cv2.imread(nombre2,0)
    
    sal = decomprimir(R,D,s,o,itera,im=dummy,largo=im.shape[0],ancho=im.shape[1], mirar=mira)
         
    for i in range(0,len(sal)):
            cv2.imwrite(nombre[:-4]+'_decomprimido_'+str(i)+'.png',sal[i])
    
    if mira==1:
    
        for i in range(0,len(sal)):    
            mostrar(im,sal[i],'original','decomprimido')
            
 
def cargar(nombre,itera):
    """
    
    carga los parametros de una imagen comprimida
    
    """
    s= np.load(nombre+'_comprimido_'+str(itera)+'_s.npy')
    o= np.load(nombre+'_comprimido_'+str(itera)+'_o.npy')
    D= np.load(nombre+'_comprimido_'+str(itera)+'_D.npy')
    R= np.load(nombre+'_comprimido_'+str(itera)+'_R.npy')
     
    return s,o,D,R
    



def zoom(n,R,D,s,o):
    """
    
    Implementacion de interpolacion fractal para zoom (por ahora se requiere que la imagen esta comprimida )
    
    """
    return decomprimir(R*n,D*n,s,o,10,0,n*64,n*64)
    


def ver(R,D,im,param=0):
    """
    
    Muestra cada elemento de R y su correspondiente enlace con un elemento de D
    
    """
    for i in range(param,R.shape[0]):
        mostrar_cuadricula(R[i],D[i],im)

    

