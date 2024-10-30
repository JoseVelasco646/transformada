import numpy as np

def mostrar_M(N):
    n = np.arange(N)
    k = n.reshape((N,1))

    M = k * n
    print("M:", M)  


def obtener_datos(longitud):
    datos = np.random.random(longitud)
    return datos 

def obtener_terminos_circulares(N):
    terminos = np.exp(-1j * 2 * np.pi * np.arange(N) / N)
    return terminos

def transformada_discreta_fourier(datos):
    # longitud de los datos
    N = datos.shape[0] 
    
    n = np.arange(N)
    k = n.reshape((N,1))
    M = np.exp(-1j * 2 * np.pi * k * n / N)
   
    return np.dot(M, datos)

def transformada_rapida_fourier(datos):
    #Obtener tama;o de los datos
    N = datos.shape[0]

    #Debe ser una potencia de 2
    assert N % 2 == 0, 'La longitud de los datos: {} debe ser una potencia de 2'.format(N)

    if N <= 2:
        return transformada_discreta_fourier(datos)

    else:
        datos_pares = transformada_rapida_fourier(datos[::2])
        datos_impares = transformada_rapida_fourier(datos[1::2])
        terminos = obtener_terminos_circulares(N)

        return np.concatenate(
            [
            datos_pares + terminos[:N//2] * datos_impares,
            datos_pares + terminos[N//2:] * datos_impares 
            ])
    

N = 4

X = obtener_datos(N)
print("Datos: ", X)

tdf = transformada_discreta_fourier(X)
trf = transformada_rapida_fourier(X)
tdfnp = np.fft.fft(X)

print('TDF:', trf)

print(np.allclose(tdf, tdfnp),
      np.allclose(trf, tdfnp))

print("")
mostrar_M(N)
