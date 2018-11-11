#ifndef BMP_H
#define BMP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned char uchar;

//gcc -o bmpV1 bmpV1.c
//./bmpV1 linux_detergente.bmp -x2
typedef struct BMP{
    char bm[2];					//(2 Bytes) BM (Tipo de archivo)
    int tamano;					//(4 Bytes) Tamaño del archivo en bytes
    int reservado;				//(4 Bytes) Reservado
    int offset;					//(4 Bytes) offset, distancia en bytes entre la img y los píxeles
    int tamanoMetadatos;		//(4 Bytes) Tamaño de Metadatos (tamaño de esta estructura = 40)
    int ancho;					//(4 Bytes) Alto (numero de pixeles verticales)
    int alto;					//(4 Bytes) Ancho (numero de píxeles horizontales)
    short int numeroPlanos;		//(2 Bytes) Numero de planos de color
    short int profundidadColor;	//(2 Bytes) Profundidad de color (debe ser 24 para nuestro caso)
    int tipoCompresion;			//(4 Bytes) Tipo de compresión (Vale 0, ya que el bmp es descomprimido)
    int tamanoEstructura;		//(4 Bytes) Tamaño de la estructura Imagen (Paleta)
	int pxmh;					//(4 Bytes) Píxeles por metro horizontal
	int pxmv;					//(4 Bytes) Píxeles por metro vertical
    int coloresUsados;			//(4 Bytes) Cantidad de colores usados
	int coloresImportantes;		//(4 Bytes) Cantidad de colores importantes
    uchar **pixelB;             //Puntero a una tabla dinamica de caracteres de 2 dimenciones almacenara el valor del pixel en escala de BLUE (0-255)
    uchar **pixelG;             //Puntero a una tabla dinamica de caracteres de 2 dimenciones almacenara el valor del pixel en escala de GREEN (0-255)
    uchar **pixelR;             //Puntero a una tabla dinamica de caracteres de 2 dimenciones almacenara el valor del pixel en escala de RED (0-255)
}BMP;

void abrir_imagen(BMP *imagen, char ruta[]);	//Función para abrir la imagen BMP
void crear_imagen(BMP *imagen, char ruta[],int escala);	//Función para crear una imagen BMP


/*
int main (int argc, char* argv[]){
	BMP img;				//Estructura de tipo imágen
	char IMAGEN[45];		//Almacenará la ruta de la imagen
	char SCALE[3];
	int escala;
		
	//Si no se introduce una ruta de imágen
	if (argc!=3) 
	{
		printf("Indique el nombre del archivo a codificar - Ejemplo: [user@equipo]$ %s imagen.bmp\n",argv[0]);
		printf("Indique el valor de la escala - Ejemplo: -x2\n");
		exit(1);
	} 
	//Almacenar la ruta de la imágen
	strcpy(IMAGEN,argv[1]);
	//Almacenar del valor de escala
	strcpy(SCALE,argv[2]);
	printf("%s\n",SCALE);
	escala=SCALE[2]-48;
	printf("%d\n",escala);

    abrir_imagen(&img,IMAGEN);
	crear_imagen(&img,IMAGEN_TRATADA,escala);

	exit (0);	
}
*/




#endif // BMP_H
