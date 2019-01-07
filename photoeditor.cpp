#include "photoeditor.h"
#include "ui_photoeditor.h"
#include <QtCore>
#include <QtGui>
#include <QFileDialog>

#include "includes.h"

#include <library/bmp.h>
#include "library/utils.h"

#include "ColorSpace/ColorSpace.h"
#include "Filter/Filter.h"
#include "Filter/kernel.h"

#include "FFT/fft.h"
#include "GeomTrans/geomTransform.h"
#include "Video/matching.h"


PhotoEditor::PhotoEditor(QWidget *parent) : QMainWindow(parent),
                                            ui(new Ui::PhotoEditor){
    ui->setupUi(this);
    setDisabledImageSection(false);
    setDisabledVideoSection(false);

    // Create Original scene
    ui->original->setScene(new QGraphicsScene(this));
    ui->original->scene()->addItem(&PixOriginal);
}

PhotoEditor::~PhotoEditor(){
    delete [] R ; delete [] G ; delete [] B ;
    delete ui;
}

void PhotoEditor::on_actionAbrir_triggered(){
    QString fileName = QFileDialog::getOpenFileName(this,
                                 tr("Abrir Imagen BMP"), "/home/victor/Imágenes/",
                                 tr("Imagen BMP (*.bmp);;Imagen (*.bmp *.png *.jpg  *.jpeg *.gif *.tif);;Video (*.mp4 *.avi *.mpg *.mpeg)"));
    pathTo = fileName.toStdString();

    // BMP format
    if( check_BMP_format(pathTo) ){
        abrir_imagen(&img,pathTo.c_str());
        setDisabledImageSection( true);
        setDisabledVideoSection(false);
    }

    // All image format
    if( checkImageformat(pathTo) ){
        abrir_imagen(R,G,B,rows,cols,pathTo);
        setDisabledImageSection( true);
        setDisabledVideoSection(false);

        cv::Mat image;
        rgb2mat(R,G,B,rows,cols,image);
        QImage qframe(image.data,image.cols,image.rows,int(image.step),QImage::Format_RGB888);
        PixOriginal.setPixmap( QPixmap::fromImage(qframe.rgbSwapped()) );
        ui->original->fitInView(&PixOriginal, Qt::KeepAspectRatio);
    }

    // All video format
    if( checkVideoformat(pathTo) ){
        video = cv::VideoCapture(pathTo);
        if (!video.isOpened()){
            QMessageBox messageBox;
            messageBox.critical(0,"Error","Failed to open the video");
            messageBox.setFixedSize(500,200);
        }
        setDisabledImageSection(false);
        setDisabledVideoSection( true);

        cv::Mat image;
        video >> image;
        QImage qframe(image.data,image.cols,image.rows,int(image.step),QImage::Format_RGB888);
        PixOriginal.setPixmap( QPixmap::fromImage(qframe.rgbSwapped()) );
        ui->original->fitInView(&PixOriginal, Qt::KeepAspectRatio);
    }
}

void PhotoEditor::on_actionGuardar_triggered(){
    QString fileName = QFileDialog::getSaveFileName(this,
                                 tr("Guardar Imagen BMP"), "",
                                 tr("Imagen BMP (*.bmp);;All Files (*)"));
    pathTo = fileName.toStdString();
    crear_imagen(&img,pathTo.c_str(),1);
}

void PhotoEditor::on_actionCMY_triggered(){
    uchar *C1 = new unsigned char[rows*cols];
    uchar *C2 = new unsigned char[rows*cols];
    uchar *C3 = new unsigned char[rows*cols];

    transformColorModel(R,G,B,C1,C2,C3,rows*cols,0);
    showColorModel(C1,C2,C3,rows,cols,"CMY");

    delete [] C1; delete [] C2; delete [] C3;
}

void PhotoEditor::on_actionHSL_triggered(){
    uchar *C1 = new unsigned char[rows*cols];
    uchar *C2 = new unsigned char[rows*cols];
    uchar *C3 = new unsigned char[rows*cols];

    transformColorModel(R,G,B,C1,C2,C3,rows*cols,1);
    showColorModel(C1,C2,C3,rows,cols,"HSL");

    delete [] C1; delete [] C2; delete [] C3;
}

void PhotoEditor::on_actionHSV_triggered(){
    uchar *C1 = new unsigned char[rows*cols];
    uchar *C2 = new unsigned char[rows*cols];
    uchar *C3 = new unsigned char[rows*cols];

    transformColorModel(R,G,B,C1,C2,C3,rows*cols,2);
    showColorModel(C1,C2,C3,rows,cols,"HSV");

    delete [] C1; delete [] C2; delete [] C3;
}

void PhotoEditor::on_actionXYZ_triggered(){
    uchar *C1 = new unsigned char[rows*cols];
    uchar *C2 = new unsigned char[rows*cols];
    uchar *C3 = new unsigned char[rows*cols];

    transformColorModel(R,G,B,C1,C2,C3,rows*cols,3);
    showColorModel(C1,C2,C3,rows,cols,"XYZ");

    delete [] C1; delete [] C2; delete [] C3;
}

void PhotoEditor::on_actionLMS_triggered(){
    uchar *C1 = new unsigned char[rows*cols];
    uchar *C2 = new unsigned char[rows*cols];
    uchar *C3 = new unsigned char[rows*cols];

    transformColorModel(R,G,B,C1,C2,C3,rows*cols,4);
    showColorModel(C1,C2,C3,rows,cols,"LMS");

    delete [] C1; delete [] C2; delete [] C3;
}

void PhotoEditor::on_actionYIQ_triggered(){
    uchar *C1 = new unsigned char[rows*cols];
    uchar *C2 = new unsigned char[rows*cols];
    uchar *C3 = new unsigned char[rows*cols];

    transformColorModel(R,G,B,C1,C2,C3,rows*cols,5);
    showColorModel(C1,C2,C3,rows,cols,"YIQ");

    delete [] C1; delete [] C2; delete [] C3;
}

void PhotoEditor::on_actionYUV_triggered(){
    uchar *C1 = new unsigned char[rows*cols];
    uchar *C2 = new unsigned char[rows*cols];
    uchar *C3 = new unsigned char[rows*cols];

    transformColorModel(R,G,B,C1,C2,C3,rows*cols,6);
    showColorModel(C1,C2,C3,rows,cols,"YUV");

    delete [] C1; delete [] C2; delete [] C3;
}

void PhotoEditor::on_actionYCbCr_triggered(){
    uchar *C1 = new unsigned char[rows*cols];
    uchar *C2 = new unsigned char[rows*cols];
    uchar *C3 = new unsigned char[rows*cols];

    transformColorModel(R,G,B,C1,C2,C3,rows*cols,7);
    showColorModel(C1,C2,C3,rows,cols,"YCbCr");

    delete [] C1; delete [] C2; delete [] C3;
}

void PhotoEditor::on_actionGaussianFilter_triggered(){
    uchar *C1 = new unsigned char[rows*cols];
    uchar *C2 = new unsigned char[rows*cols];
    uchar *C3 = new unsigned char[rows*cols];

    Filter blur(GAUSS_KERNEL);

    for(int q=0 ; q<5;++q){
        if(q == 0){
            blur.convolution(R,C1,rows,cols,4);
            blur.convolution(G,C2,rows,cols,4);
            blur.convolution(B,C3,rows,cols,4);
        }
        else{
            blur.convolution(C1,C1,rows,cols,4);
            blur.convolution(C2,C2,rows,cols,4);
            blur.convolution(C3,C3,rows,cols,4);
        }
    }

    showColorImage(C1,C2,C3,rows,cols,"Gaussian Filter");
}

void PhotoEditor::on_actionLaplaceFilter_triggered(){
    uchar *C1 = new unsigned char[rows*cols];
    uchar *C2 = new unsigned char[rows*cols];
    uchar *C3 = new unsigned char[rows*cols];

    Filter blur(LAPLACE_KERNEL);

    for(int q=0 ; q<5;++q){
        if(q == 0){
            blur.convolution(R,C1,rows,cols,4);
            blur.convolution(G,C2,rows,cols,4);
            blur.convolution(B,C3,rows,cols,4);
        }
        else{
            blur.convolution(C1,C1,rows,cols,4);
            blur.convolution(C2,C2,rows,cols,4);
            blur.convolution(C3,C3,rows,cols,4);
        }
    }

    showColorImage(C1,C2,C3,rows,cols,"Gaussian Filter");
}

void PhotoEditor::on_actionGabor0Filter_triggered(){
    uchar *C1 = new unsigned char[rows*cols];
    uchar *C2 = new unsigned char[rows*cols];
    uchar *C3 = new unsigned char[rows*cols];

    Filter blur(GABOR_00_KERNEL);

    blur.convolution(R,C1,rows,cols,4);
    blur.convolution(G,C2,rows,cols,4);
    blur.convolution(B,C3,rows,cols,4);

    showColorImage(C1,C2,C3,rows,cols,"Gabor 0° Filter");
}

void PhotoEditor::on_actionGabor45Filter_triggered(){
    uchar *C1 = new unsigned char[rows*cols];
    uchar *C2 = new unsigned char[rows*cols];
    uchar *C3 = new unsigned char[rows*cols];

    Filter blur(GABOR_45_KERNEL);

    blur.convolution(R,C1,rows,cols,4);
    blur.convolution(G,C2,rows,cols,4);
    blur.convolution(B,C3,rows,cols,4);

    showColorImage(C1,C2,C3,rows,cols,"Gabor 45° Filter");
}

void PhotoEditor::on_actionGabor90Filter_triggered(){
    uchar *C1 = new unsigned char[rows*cols];
    uchar *C2 = new unsigned char[rows*cols];
    uchar *C3 = new unsigned char[rows*cols];

    Filter blur(GABOR_90_KERNEL);

    blur.convolution(R,C1,rows,cols,4);
    blur.convolution(G,C2,rows,cols,4);
    blur.convolution(B,C3,rows,cols,4);

    showColorImage(C1,C2,C3,rows,cols,"Gabor 90° Filter");
}

void PhotoEditor::on_actionGabor135Filter_triggered(){
    uchar *C1 = new unsigned char[rows*cols];
    uchar *C2 = new unsigned char[rows*cols];
    uchar *C3 = new unsigned char[rows*cols];

    Filter blur(GABOR_135_KERNEL);

    blur.convolution(R,C1,rows,cols,4);
    blur.convolution(G,C2,rows,cols,4);
    blur.convolution(B,C3,rows,cols,4);

    showColorImage(C1,C2,C3,rows,cols,"Gabor 135° Filter");
}

void PhotoEditor::setDisabledImageSection(const bool &_v){
    // Model color Enabled
    ui->actionCMY  ->setEnabled(_v);
    ui->actionHSL  ->setEnabled(_v);
    ui->actionHSV  ->setEnabled(_v);
    ui->actionLMS  ->setEnabled(_v);
    ui->actionXYZ  ->setEnabled(_v);
    ui->actionYIQ  ->setEnabled(_v);
    ui->actionYUV  ->setEnabled(_v);
    ui->actionYCbCr->setEnabled(_v);

    // Filter Enabled
    ui->actionGabor0Filter  ->setEnabled(_v);
    ui->actionGabor45Filter ->setEnabled(_v);
    ui->actionGabor90Filter ->setEnabled(_v);
    ui->actionLaplaceFilter ->setEnabled(_v);
    ui->actionGabor135Filter->setEnabled(_v);
    ui->actionGaussianFilter->setEnabled(_v);

    // FFT Enabled
    ui->actionPhaseFFT    ->setEnabled(_v);
    ui->actionMagnitudeFFT->setEnabled(_v);

    // Geometric Transformation Enabled
    ui->actionPerspective->setEnabled(_v);
}

void PhotoEditor::setDisabledVideoSection(const bool &_v){
    ui->actionTrackingVideo->setEnabled(_v);
}


void PhotoEditor::on_actionMagnitudeFFT_triggered(){
    uchar *C1 = new unsigned char[rows*cols];
    calculateMagnitudeFFT(R,G,B,C1, uint(rows), uint(cols));

    cv::Mat out = cv::Mat(rows,cols,CV_8UC1,C1);
    cv::namedWindow( "Magnitude FFT" );
    cv::imshow( "Magnitude FFT", out );

    delete [] C1;
}

void PhotoEditor::on_actionPhaseFFT_triggered(){
    uchar *C1 = new unsigned char[rows*cols];
    calculatePhaseFFT(R,G,B,C1, uint(rows), uint(cols));

    cv::Mat out = cv::Mat(rows,cols,CV_8UC1,C1);
    cv::namedWindow( "Phase FFT" );
    cv::imshow( "Phase FFT", out );

    delete [] C1;
}

void PhotoEditor::on_actionPerspective_triggered(){
    geometricTransformation(R,G,B, uint(rows), uint(cols));
}

void PhotoEditor::on_actionTrackingVideo_triggered(){
    matching(video);
}
