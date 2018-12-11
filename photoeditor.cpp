#include "photoeditor.h"
#include "ui_photoeditor.h"
#include <QtCore>
#include <QtGui>
#include <QFileDialog>
#include <library/bmp.h>

#include "ColorSpace/ColorSpace.h"
#include "Filter/Filter.h"
#include "Filter/kernel.h"

#include "FFT/fft.h"

#include<opencv2/core/core.hpp>
#include<opencv2/ml/ml.hpp>
#include<opencv/cv.h>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/video/background_segm.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/imgcodecs.hpp>

#include <cufft.h>

PhotoEditor::PhotoEditor(QWidget *parent) : QMainWindow(parent),
                                            ui(new Ui::PhotoEditor){
    ui->setupUi(this);
    //scene        = new QGraphicsScene(this);
    //graphicsView = new QGraphicsView (this);

    //scene->setSceneRect( QRectF(2,2,800,800) );
}

PhotoEditor::~PhotoEditor(){
    delete ui;
}

void PhotoEditor::on_actionAbrir_triggered(){
    QString fileName = QFileDialog::getOpenFileName(this,
                                 tr("Abrir Imagen BMP"), "",
                                 tr("Imagen BMP (*.bmp);;All Files (*)"));
    QByteArray ba = fileName.toLocal8Bit();
    abrir_imagen(&img,ba.data());

    QImage image(img.pixel, img.ancho, img.alto, QImage::Format_RGB888);
}

void PhotoEditor::on_actionGuardar_triggered(){
    QString fileName = QFileDialog::getSaveFileName(this,
                                 tr("Guardar Imagen BMP"), "",
                                 tr("Imagen BMP (*.bmp);;All Files (*)"));
    QByteArray ba = fileName.toLocal8Bit();
    crear_imagen(&img,ba.data(),1);
}

void PhotoEditor::on_pushButton_clicked(){

    // Read image
    cv::Mat img = cv::imread("/home/victor/Documentos/Imagenes/PhotoEditor/lena.jpg");
    int rows = img.rows;
    int cols = img.cols;

    /*
    cv::namedWindow( "Image" );
    cv::imshow( "Image", img );

    cv::waitKey(0);
    */
    std::cout << "size: (" << rows << "," << cols  << ")"<< std::endl;

    unsigned char *R = new unsigned char[rows*cols];
    unsigned char *G = new unsigned char[rows*cols];
    unsigned char *B = new unsigned char[rows*cols];

    // Getting data
    int id = 0;
    for(int i=0; i<rows; i++) for(int j=0; j<cols; j++){
        B[id] = img.at<cv::Vec3b>(i,j)[0];
        G[id] = img.at<cv::Vec3b>(i,j)[1];
        R[id] = img.at<cv::Vec3b>(i,j)[2];
        ++id;
    }

    unsigned char *C1 = new unsigned char[rows*cols];
    unsigned char *C2 = new unsigned char[rows*cols];
    unsigned char *C3 = new unsigned char[rows*cols];


/*
 *  Modelos de color:
 *  ----------------
 *    - 0: CMY       - 4: LMS
 *    - 1: HSL       - 5: YIQ
 *    - 2: HSV       - 6: YUV
 *    - 3: XYZ       - 7: YCbCr
 */
/*
    transformColorModel(R,G,B,C1,C2,C3,rows*cols,2);

    cv::Mat out = cv::Mat(rows,cols,CV_8UC1,C1);
    cv::normalize(out, out, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    std::cout << "Channel 1: (" << out.rows << "," << out.cols  << ")"<< std::endl;
    cv::namedWindow( "Channel 1" );
    cv::imshow( "Channel 1", out );

    out = cv::Mat(rows,cols,CV_8UC1,C2);
    cv::normalize(out, out, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::namedWindow( "Channel 2" );
    cv::imshow( "Channel 2", out );

    out = cv::Mat(rows,cols,CV_8UC1,C3);
    cv::normalize(out, out, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::namedWindow( "Channel 3" );
    cv::imshow( "Channel 3", out );

    cv::waitKey(0);
*/
/*
 *  Filtrado/Convolucion
 *  --------------------
 */
/*
    Filter blur(LAPLACE_KERNEL);

    for(int q=0 ; q<5;++q){
        blur.convolution(R,R,rows,cols,4);
        blur.convolution(G,G,rows,cols,4);
        blur.convolution(B,B,rows,cols,4);
    }

    std::vector<cv::Mat> array_to_merge;
    array_to_merge.push_back(cv::Mat(rows,cols,CV_8UC1,B));
    array_to_merge.push_back(cv::Mat(rows,cols,CV_8UC1,G));
    array_to_merge.push_back(cv::Mat(rows,cols,CV_8UC1,R));

    cv::Mat out;
    cv::merge(array_to_merge, out);

    cv::namedWindow( "New Image" );
    cv::imshow( "New Image", out );

    cv::waitKey(0);
*/

    executeFFT(R,G,B,C1, uint(rows), uint(cols));

    cv::Mat out = cv::Mat(rows,cols,CV_8UC1,C1);
    cv::namedWindow( "FFT" );
    cv::imshow( "FFT", out );


    // Delete
    delete [] R ; delete [] G ; delete [] B ;
    delete [] C1; delete [] C2; delete [] C3;

}
