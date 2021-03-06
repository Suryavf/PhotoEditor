#include <library/bmp.h>

//*************************************************************************************************************************************************
// Función para abrir la imagen, colocarla en escala de grisis en la estructura imagen imagen (Arreglo de bytes de alto*ancho  --- 1 Byte por pixel 0-255)
// Parametros de entrada: Referencia a un BMP (Estructura BMP), Referencia a la cadena ruta char ruta[]=char *ruta
// Parametro que devuelve: Ninguno
//*************************************************************************************************************************************************
void abrir_imagen(BMP *imagen, const char *ruta){
    FILE *archivo;	//Puntero FILE para el archivo de imágen a abrir

    //Abrir el archivo de imágen
    archivo = fopen( ruta, "rb+" );
    if(!archivo){
        //Si la imágen no se encuentra en la ruta dada
        printf( "La imágen %s no se encontro\n",ruta);
        exit(1);
    }

    //Leer la cabecera de la imagen y almacenarla en la estructura a la que apunta imagen
    fseek( archivo,0, SEEK_SET);
    fread(&imagen->bm,sizeof(char),2, archivo);
    fread(&imagen->tamano,sizeof(int),1, archivo);
    fread(&imagen->reservado,sizeof(int),1, archivo);
    fread(&imagen->offset,sizeof(int),1, archivo);
    fread(&imagen->tamanoMetadatos,sizeof(int),1, archivo);
    fread(&imagen->ancho,sizeof(int),1, archivo);
    fread(&imagen->alto,sizeof(int),1, archivo);
    fread(&imagen->numeroPlanos,sizeof(short int),1, archivo);
    fread(&imagen->profundidadColor,sizeof(short int),1, archivo);
    fread(&imagen->tipoCompresion,sizeof(int),1, archivo);
    fread(&imagen->tamanoEstructura,sizeof(int),1, archivo);
    fread(&imagen->pxmh,sizeof(int),1, archivo);
    fread(&imagen->pxmv,sizeof(int),1, archivo);
    fread(&imagen->coloresUsados,sizeof(int),1, archivo);
    fread(&imagen->coloresImportantes,sizeof(int),1, archivo);

    //Validar ciertos datos de la cabecera de la imágen
    if (imagen->bm[0]!='B'||imagen->bm[1]!='M')	{
        printf ("La imagen debe ser un bitmap.\n");
        exit(1);
    }
    if (imagen->profundidadColor!= 24) {
        printf ("La imagen debe ser de 24 bits.\n");
        exit(1);
    }
    int size = imagen->alto*imagen->ancho;
    imagen->pixel = new uchar[size*3];

    uchar R,B,G;
    int id = 0;
    for (int p = 0; p<size; ++p){
        fread(&B,sizeof(char),1, archivo);  //Byte Blue del pixel
        fread(&G,sizeof(char),1, archivo);  //Byte Green del pixel
        fread(&R,sizeof(char),1, archivo);  //Byte Red del pixel

        imagen->pixel[id] = B; ++id;
        imagen->pixel[id] = G; ++id;
        imagen->pixel[id] = R; ++id;
    }

    //Cerrrar el archivo
    fclose(archivo);
}

//****************************************************************************************************************************************************
// Función para crear una imagen BMP, a partir de la estructura imagen imagen (Arreglo de bytes de alto*ancho  --- 1 Byte por pixel 0-255)
// Parametros de entrada: Referencia a un BMP (Estructura BMP), Referencia a la cadena ruta char ruta[]=char *ruta
// Parametro que devuelve: Ninguno
//****************************************************************************************************************************************************
void crear_imagen(BMP *imagen, const char ruta[],int escala){
    FILE *archivo;	//Puntero FILE para el archivo de imágen a abrir

    int i,j;

    //Abrir el archivo de imágen
    archivo = fopen( ruta, "wb+" );
    if(!archivo){
        //Si la imágen no se encuentra en la ruta dada
        printf( "La imágen %s no se pudo crear\n",ruta);
        exit(1);
    }
    //nueva img contenida en nueva estructura
    BMP imgNew;
    imgNew.bm[0]=imagen->bm[0];
    imgNew.bm[1]=imagen->bm[1];

    imgNew.reservado=imagen->reservado;
    imgNew.offset=imagen->offset;
    imgNew.tamanoMetadatos=imagen->tamanoMetadatos;

    imgNew.numeroPlanos=imagen->numeroPlanos;
    imgNew.profundidadColor=imagen->profundidadColor;
    imgNew.tipoCompresion=imagen->tipoCompresion;

    imgNew.pxmh=imagen->pxmh;
    imgNew.pxmv=imagen->pxmv;
    imgNew.coloresUsados=imagen->coloresUsados;
    imgNew.coloresImportantes=imagen->coloresImportantes;

    imgNew.alto=(imagen->alto)*escala;
    imgNew.ancho=(imagen->ancho)*escala;
    imgNew.tamanoEstructura=imgNew.alto*imgNew.ancho*3;
    imgNew.tamano=54+imgNew.tamanoEstructura;


    ////////////////////////////////////////////////////////////
    imgNew.pixelB = new uchar*[imgNew.alto];
    imgNew.pixelG = new uchar*[imgNew.alto];
    imgNew.pixelR = new uchar*[imgNew.alto];

    for(i=0;i<imgNew.alto;++i) imgNew.pixelB[i] = new uchar[imgNew.ancho];
    for(i=0;i<imgNew.alto;++i) imgNew.pixelG[i] = new uchar[imgNew.ancho];
    for(i=0;i<imgNew.alto;++i) imgNew.pixelR[i] = new uchar[imgNew.ancho];

    //Pasar la imágen al arreglo reservado en escala de BLUE,GREEN y RED
    for (i=0;i<imgNew.alto;i++){
        for (j=0;j<imgNew.ancho;j++){
            imgNew.pixelB[i][j]=(0);
            imgNew.pixelG[i][j]=(0);
            imgNew.pixelR[i][j]=(0);
        }
    }

    //paso de pixeles imagen a posiciones pares de imgNew
    for (i=0;i<imgNew.alto;i++){
        for (j=0;j<imgNew.ancho;j++){
            if((i%escala==0) && (j%escala==0)){
                imgNew.pixelB[i][j]=(imagen->pixelB[i/escala][j/escala]);
                imgNew.pixelG[i][j]=(imagen->pixelG[i/escala][j/escala]);
                imgNew.pixelR[i][j]=(imagen->pixelR[i/escala][j/escala]);
            }
        }
    }

    //interpolación horizontal de pixeles
    for (i=0;i<imgNew.alto;i++){
        for (j=1;j<imgNew.ancho;j++){
            if((i%escala==0) && (j%escala!=0)){
                imgNew.pixelB[i][j]=(imgNew.pixelB[i][j-1]+imgNew.pixelB[i][j+1])/2;
                imgNew.pixelG[i][j]=(imgNew.pixelG[i][j-1]+imgNew.pixelG[i][j+1])/2;
                imgNew.pixelR[i][j]=(imgNew.pixelR[i][j-1]+imgNew.pixelR[i][j+1])/2;
            }
        }
    }

    //interpolación vertical de pixeles
    for (i=0;i<imgNew.alto-1;i++){
        for (j=0;j<imgNew.ancho;j++){
            if(i%escala!=0){
                imgNew.pixelB[i][j]=(imgNew.pixelB[i-1][j]+imgNew.pixelB[i+1][j])/2;
                imgNew.pixelG[i][j]=(imgNew.pixelG[i-1][j]+imgNew.pixelG[i+1][j])/2;
                imgNew.pixelR[i][j]=(imgNew.pixelR[i-1][j]+imgNew.pixelR[i+1][j])/2;
            }
        }
    }

    //Escribir la cabecera de la imagen en el archivo
    fseek( archivo,0, SEEK_SET);
    fwrite(&imgNew.bm,sizeof(char),2, archivo);
    fwrite(&imgNew.tamano,sizeof(int),1, archivo);
    fwrite(&imgNew.reservado,sizeof(int),1, archivo);
    fwrite(&imgNew.offset,sizeof(int),1, archivo);
    fwrite(&imgNew.tamanoMetadatos,sizeof(int),1, archivo);
    fwrite(&imgNew.ancho,sizeof(int),1, archivo);
    fwrite(&imgNew.alto,sizeof(int),1, archivo);

    fwrite(&imgNew.numeroPlanos,sizeof(short int),1, archivo);
    fwrite(&imgNew.profundidadColor,sizeof(short int),1, archivo);
    fwrite(&imgNew.tipoCompresion,sizeof(int),1, archivo);
    fwrite(&imgNew.tamanoEstructura,sizeof(int),1, archivo);
    fwrite(&imgNew.pxmh,sizeof(int),1, archivo);
    fwrite(&imgNew.pxmv,sizeof(int),1, archivo);
    fwrite(&imgNew.coloresUsados,sizeof(int),1, archivo);
    fwrite(&imgNew.coloresImportantes,sizeof(int),1, archivo);

    //Pasar la imágen del arreglo reservado en escala de BLUE,GREEN y RED a el archivo (Deben escribirse los valores BGR)
    for (i=0;i<imgNew.alto/*imagen->alto*/;i++){
        for (j=0;j<imgNew.ancho/*imagen->ancho*/;j++){
            // Ecribir los 3 bytes BGR al archivo BMP, en este caso todos se igualan al mismo valor
            // (Valor del pixel en la matriz de la estructura imagen)
            fwrite(&imgNew.pixelB[i][j],sizeof(char),1, archivo);  //Escribir el Byte Blue del pixel
            fwrite(&imgNew.pixelG[i][j],sizeof(char),1, archivo);  //Escribir el Byte Green del pixel
            fwrite(&imgNew.pixelR[i][j],sizeof(char),1, archivo);  //Escribir el Byte Red del pixel
        }
    }

    //Cerrrar el archivo
    fclose(archivo);
}
