from PIL import Image
import numpy as np

# Extracción de los pixeles de una imagen

def fuente_info(imagen):
    '''Una función que simula una fuente de
    información al importar una imagen y 
    retornar un vector de NumPy con las 
    dimensiones de la imagen, incluidos los
    canales RGB: alto x largo x 3 canales

    :param imagen: Una imagen en formato JPG
    :return: un vector de pixeles
    '''
    img = Image.open(imagen)
    
    return np.array(img)



# Codificación de pixeles a una base binaria 
def rgb_a_bit(array_imagen):
    '''Convierte los pixeles de base 
    decimal (de 0 a 255) a binaria 
    (de 00000000 a 11111111).

    :param imagen: array de una imagen 
    :return: Un vector de (1 x k) bits 'int'
    '''
    # Obtener las dimensiones de la imagen
    x, y, z = array_imagen.shape
    
    # Número total de elementos (pixeles x canales)
    n_elementos = x * y * z

    # Convertir la imagen a un vector unidimensional de n_elementos
    pixeles = np.reshape(array_imagen, n_elementos)

    # Convertir los canales a base 2
    bits = [format(pixel, '08b') for pixel in pixeles]
    bits_Rx = np.array(list(''.join(bits)))
    
    return bits_Rx.astype(int)

# Modulación para 8-PSK
def modulador(bits, fc, mpp):
    '''Un método que simula el esquema de
       modulación digital 8-PSK.
       :param bits: Vector unidimensional de bits
       :param fc: Frecuencia de la portadora en Hz
       :param mpp: Cantidad de muestras por periodo de onda portadora
       :return: Un vector con la señal modulada
       :return: Un valor con la potencia promedio [W]
       :return: La onda portadora c(t)
       :return: La onda cuadrada moduladora (información)
    '''
# 1. Parámetros de la 'señal' de información (bits)
    N = len(bits) # Cantidad de bits
    
#2. Construyendo el periodo de la señal portadora
    Tc = 1 / fc  # periodo [s]
    t_periodo = np.linspace(0, Tc, mpp)  # mpp: muestras por período
    
# Portadora 1 de s(t)
    portadora1 = np.cos(2*np.pi*fc*t_periodo)
    
# Portadora 2 de s(t)
    portadora2 = np.sin(2*np.pi*fc*t_periodo)
    
# 3. Inicializar la señal modulada s(t)
    t_simulacion = np.linspace(0, N*Tc, N*mpp) 
    senal_Tx = np.zeros(t_simulacion.shape)
    moduladora1 = np.zeros(t_simulacion.shape)  # (opcional) señal de bits
    moduladora2 = np.zeros(t_simulacion.shape)  # (opcional) señal de bits
    # Se define h
    h = np.sqrt(2)/2
    
# 4. Asignar las formas de onda según los bits (8-PSK)
    for i in range(0, len(bits), 3):
            # Para este tipo de modulación hay 3 bits y ocho símbolos posibles
            if(bits[i] == 1 and bits[i+1] == 1 and bits[i+2] == 1):             # 111
                senal_Tx[i*mpp:(i+1)*mpp] = portadora1 * 1 + portadora2 * 0    
                moduladora1[i*mpp:(i+1)*mpp] = 1
                moduladora2[i*mpp:(i+1)*mpp] = 0
            elif (bits[i] == 1 and bits[i+1] == 1 and bits[i+2] == 0):          # 110
                senal_Tx[i*mpp:(i+1)*mpp] = portadora1 * h + portadora2 * h
                moduladora1[i*mpp:(i+1)*mpp] = h
                moduladora2[i*mpp:(i+1)*mpp] = h
            elif (bits[i] == 0 and bits[i+1] == 1 and bits[i+2] == 0):          # 010
                senal_Tx[i*mpp:(i+1)*mpp] = portadora1 * 0 + portadora2 * 1
                moduladora1[i*mpp:(i+1)*mpp] = 0
                moduladora2[i*mpp:(i+1)*mpp] = 1
            elif (bits[i] == 0 and bits[i+1] == 1 and bits[i+2] == 1):          # 011
                senal_Tx[i*mpp:(i+1)*mpp] = portadora1 * (-1 * h) + portadora2 * h
                moduladora1[i*mpp:(i+1)*mpp] = (-1 * h)
                moduladora2[i*mpp:(i+1)*mpp] = h
            elif (bits[i] == 0 and bits[i+1] == 0 and bits[i+2] == 1):          # 001
                senal_Tx[i*mpp:(i+1)*mpp] = portadora1 * -1 + portadora2 * 0
                moduladora1[i*mpp:(i+1)*mpp] = -1
                moduladora2[i*mpp:(i+1)*mpp] = 0
            elif (bits[i] == 0 and bits[i+1] == 0 and bits[i+2] == 0):          # 000
                senal_Tx[i*mpp:(i+1)*mpp] = portadora1 * (-1 * h) + portadora2 * (-1 * h)
                moduladora1[i*mpp:(i+1)*mpp] = (-1 * h)
                moduladora2[i*mpp:(i+1)*mpp] = (-1 * h)
            elif (bits[i] == 1 and bits[i+1] == 0 and bits[i+2] == 0):          # 100,
                senal_Tx[i*mpp:(i+1)*mpp] = portadora1 * 0 + portadora2 * -1
                moduladora1[i*mpp:(i+1)*mpp] = 0
                moduladora2[i*mpp:(i+1)*mpp] = -1
            else:                                                               # 101
                senal_Tx[i*mpp:(i+1)*mpp] = portadora1 * h + portadora2 * (-1 * h)
                moduladora1[i*mpp:(i+1)*mpp] = h
                moduladora2[i*mpp:(i+1)*mpp] = (-1 * h)
    # 5. Calcular la potencia promedio de la señal modulada\n",
    P_senal_Tx = (1 / (N*Tc)) * np.trapz(pow(senal_Tx, 2), t_simulacion)
    portadoraT = portadora1 + portadora2
        
    return senal_Tx, P_senal_Tx, portadora1, portadora2, moduladora1, moduladora2

# Creación de un canal con ruido AWGN
def canal_ruidoso(senal_Tx, Pm, SNR):
    '''Un bloque que simula un medio de trans-
    misión no ideal (ruidoso) empleando ruido
    AWGN. Pide por parámetro un vector con la
    señal provieniente de un modulador y un
    valor en decibelios para la relación señal
    a ruido.

    :param senal_Tx: El vector del modulador
    :param Pm: Potencia de la señal modulada
    :param SNR: Relación señal-a-ruido en dB
    :return: La señal modulada al dejar el canal
    '''
    # Potencia del ruido generado por el canal
    Pn = Pm / pow(10, SNR/10)

    # Generando ruido auditivo blanco gaussiano (potencia = varianza)
    ruido = np.random.normal(0, np.sqrt(Pn), senal_Tx.shape)

    # Señal distorsionada por el canal ruidoso
    senal_Rx = senal_Tx + ruido

    return senal_Rx

# Demodulación para 8-PSK
def demodulador(senal_Rx, portadora1, portadora2, mpp):
    '''Un método que simula un bloque demodulador
    de señales, bajo un esquema 8-PSK. El criterio
    de demodulación se basa en decodificación por 
    detección de energía.

    :param senal_Rx: La señal recibida del canal
    :param portadora1: La onda portadora c(t)
    :param portadora2: La onda portadora2 c(t)
    :param mpp: Número de muestras por periodo
    :return: Los bits de la señal demodulada1 y 2
    '''
    # Cantidad de muestras en senal_Rx
    M = len(senal_Rx)

    # Cantidad de bits (símbolos) en transmisión
    N = int(M / mpp)

    # Vector para bits obtenidos por la demodulación
    bits_Rx = np.zeros(N)

     # Vector para la señal demodulada
    senal_demodulada1 = np.zeros(senal_Rx.shape)
    senal_demodulada2 = np.zeros(senal_Rx.shape)
    
    # Pseudo-energía de un período de la portadora
    Es1 = np.sum(portadora1 * portadora1)
    Es2 = np.sum(portadora2 * portadora2)
    
    
    
    # Demodulación
    for i in range(N):
        # Producto interno de dos funciones
        producto1 = senal_Rx[i*mpp : (i+1)*mpp] * portadora1
        Ep1 = np.sum(producto1) 
        senal_demodulada1[i*mpp : (i+1)*mpp] = producto1
        
        producto2 = senal_Rx[i*mpp : (i+1)*mpp] * portadora2
        Ep2 = np.sum(producto2) 
        senal_demodulada2[i*mpp : (i+1)*mpp] = producto2
        
        # Se define h 
        h = np.sqrt(2)/2

        # Criterio de decisión por detección de energía
        if i % 3 == 0:
            if  Ep1 >= (1+h)*Es1/2 and Ep2 >=(-1*h)*Es2/2 and Ep2 <= h*Es2/2:
                bits_Rx[i] = 1
                bits_Rx[i+1] = 1
                bits_Rx[i+2] = 1
            elif Ep1 >= h*Es1/2 and Ep1 <= (1+h)*Es1/2  and Ep2 >= h*Es2/2 and Ep2 <= (1+h)*Es2/2:
                bits_Rx[i] = 1
                bits_Rx[i+1] = 1
                bits_Rx[i+2] = 0
            elif Ep1 >= (-1*h)*Es1/2 and  Ep1 <= h*Es1/2  and Ep2 >= (1+h)*Es2/2:
                bits_Rx[i] = 0
                bits_Rx[i+1] = 1
                bits_Rx[i+2] = 0
            elif Ep1 >= (-1*(h+1))*Es1/2 and Ep1 <= (-1)*h*Es1/2 and Ep2 >= h*Es2/2 and Ep2 <= (1+h)*Es2/2:
                bits_Rx[i] = 0
                bits_Rx[i+1] = 1
                bits_Rx[i+2] = 1
            elif Ep1 <= (-1*(1+h))*Es1/2 and Ep2 >= (-1*h)*Es2/2 and Ep2 <= h*Es2/2:
                bits_Rx[i] = 0
                bits_Rx[i+1] = 0
                bits_Rx[i+2] = 1
            elif  Ep1 >= (-1*(h+1))*Es1/2 and Ep1 <= (-1)*h*Es1/2  and Ep2>= (-1*(h+1))*Es2/2 and Ep2 <= (-1)*h*Es2/2:
                bits_Rx[i] = 0
                bits_Rx[i+1] = 0
                bits_Rx[i+2] = 0
            elif Ep1 >= (-1*h)*Es1/2 and Ep1 <= h*Es1/2  and Ep2 <= (-1)*(1+h)*Es2/2 :
                bits_Rx[i] = 1
                bits_Rx[i+1] = 0
                bits_Rx[i+2] = 0
            else:
                bits_Rx[i] = 1
                bits_Rx[i+1] = 0
                bits_Rx[i+2] = 1
                
    return bits_Rx.astype(int), senal_demodulada1, senal_demodulada2

# Reconstrucción de la imagen
def bits_a_rgb(bits_Rx, dimensiones):
    '''Un blque que decodifica el los bits
    recuperados en el proceso de demodulación

    :param: Un vector de bits 1 x k 
    :param dimensiones: Tupla con dimensiones de la img.
    :return: Un array con los pixeles reconstruidos
    '''
    # Cantidad de bits
    N = len(bits_Rx)

    # Se reconstruyen los canales RGB
    bits = np.split(bits_Rx, N / 8)

    # Se decofican los canales:
    canales = [int(''.join(map(str, canal)), 2) for canal in bits]
    pixeles = np.reshape(canales, dimensiones)

    return pixeles.astype(np.uint8)

import numpy as np
import matplotlib.pyplot as plt
import time

def modulación_8PSK(relacion):
    ''' Esta función es la encargada de ejercutar desde la modulación
    hasta la demodulación de la modulación 8-PSK, llamando 
    los métodos anteriores, y fuente_info, reg_a_bit y canal_ruidoso que
    fueron dadas previamente.'''
    
    # Parámetros
    fc = 5000  # frecuencia de la portadora
    mpp = 20   # muestras por periodo de la portadora
    SNR = relacion   # relación señal-a-ruido del canal

    # Iniciar medición del tiempo de simulación
    inicio = time.time()

    # 1. Importar y convertir la imagen a trasmitir
    imagen_Tx = fuente_info('arenal.jpg')
    dimensiones = imagen_Tx.shape

    # 2. Codificar los pixeles de la imagen
    bits_Tx = rgb_a_bit(imagen_Tx)

    # 3. Modular la cadena de bits usando el esquema BPSK
    senal_Tx, Pm, portadora1, portadora2, moduladora1, moduladora2 = modulador(bits_Tx, fc, mpp)

    # 4. Se transmite la señal modulada, por un canal ruidoso
    senal_Rx = canal_ruidoso(senal_Tx, Pm, SNR)

    # 5. Se desmodula la señal recibida del canal
    bits_Rx, senal_demodulada1, senal_demodulada2 = demodulador(senal_Rx, portadora1, portadora2, mpp)

    # 6. Se visualiza la imagen recibida 
    imagen_Rx = bits_a_rgb(bits_Rx, dimensiones)
    Fig = plt.figure(figsize=(10,6))

    # Cálculo del tiempo de simulación
    print('Duración de la simulación: ', time.time() - inicio)

    # 7. Calcular número de errores
    errores = sum(abs(bits_Tx - bits_Rx))
    BER = errores/len(bits_Tx)
    print('{} errores, para un BER de {:0.4f}.'.format(errores, BER))

    # Mostrar imagen transmitida
    ax = Fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(imagen_Tx)
    ax.set_title('Transmitido')

    # Mostrar imagen recuperada
    ax = Fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(imagen_Rx)
    ax.set_title('Recuperado')
    Fig.tight_layout()

    plt.imshow(imagen_Rx)
    
    # Visualizar el cambio entre las señales
    fig, (ax2, ax3, ax4) = plt.subplots(nrows=3, sharex=True, figsize=(14, 7))

    # La señal modulada por 8-PSK
    ax2.plot(senal_Tx[0:600], color='g', lw=2) 
    ax2.set_ylabel('$s(t)$')
    
    ax2.set_title('Resultado para un SNR de {} dB'.format(SNR), fontsize = 14)

    # La señal modulada al dejar el canal
    ax3.plot(senal_Rx[0:600], color='b', lw=2) 
    ax3.set_ylabel('$s(t) + n(t)$')

    # La señal demodulada
    ax4.plot(senal_demodulada[0:600], color='m', lw=2) 
    ax4.set_ylabel('$b^{\prime}(t)$')
    ax4.set_xlabel('$t$ / milisegundos')
    fig.tight_layout()
    plt.show()
    
    return(senal_Tx)

# Se obtiene la respuesta de la parte 4.1 para un SNR = 0
senal_Tx = modulación_8PSK(0)

# Se obtiene la respuesta de la parte 4.1 para un SNR = 5
senal_Tx = modulación_8PSK(5)

# Se obtiene la respuesta de la parte 4.1 para un SNR = 20
senal_Tx = modulación_8PSK(20)

#4.2 Estacionaridad y ergodicidad

#Se debe calcular el promedio del tiempo de los 4 repeticiones en el punto t
#Vector de tiempo de la muestra
Muestra = np.linspace(0, 2, 100)

# Se define h
h= np.sqrt(2)/2

#Se fijan los valores de A1 y A2
A1 = [1, h, 0, -h, -1, -h, 0, h]
A2 = [0, h, 1, h, 0, -h, -1, -h]

# De acuerdo a la modulación 8-SPK hay 8 posibilidades
n = 8 

# Debido a la cantidad de muestras se genera una matriz vacía primero
matriz = np.empty((n,len(Muestra)))

#Se realiza la codificación para cada función
for i in range(0, len(A1)): 
    # Recorre el tamaño de A1 y es igual al de A2
    x = A1[i] * np.cos(2 * (np.pi) * fc * Muestra) + A2[i] * np.sin(2 * (np.pi) * fc * Muestra)
    # Se guardan los valores en la matriz 
    matriz[i,:] = x
    plt.plot(Muestra, x, lw = 2)

    
#Se calcula el promedio temporal obtenido de las muestras realizadaas
PromTemp =  [np.mean(matriz[:,i]) for i in range(len(Muestra))]
plt.plot(Muestra, PromTemp, color = 'm', lw = 7, label = 'Promedio Temporal')

                  
#Ahora se debe calcular el promedio del promedio estadístico o teórico
PromEst =  np.mean(senal_Tx) * Muestra
plt.plot(Muestra, PromEst, color = 'y', lw = 2, label = 'Promedio Estadístico')


plt.xlabel('$t$')
plt.ylabel('$X(t)$')
plt.title('Promedio temporal y estadístico de la señal $X(t)$')
plt.show()

# 4.3 Densidad espectral de potencia 
from scipy.fftpack import fft

# Transformada de Fourier
senal_f = fft(senal_Tx)

# Muestras de la señal
Nm = len(senal_Tx)

# Número de símbolos
Ns = Nm // mpp

# Tiempo del símbolo = periodo de la onda portadora
Tc = 1 / fc

# Tiempo entre muestras (período de muestreo)
Tm = Tc / mpp

# Tiempo de la simulación
T = Ns * Tc

# Espacio de frecuencias
f = np.linspace(0.0, 1.0/(2.0*Tm), Nm//2)

# Gráfica
plt.plot(f, 2.0/Nm * np.power(np.abs(senal_f[0:Nm//2]), 2))
plt.xlim(0, 20000)
plt.grid()
plt.title('Densidad espectral de potencia para la señal modulada')
plt.show()