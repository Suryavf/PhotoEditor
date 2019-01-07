#ifndef PHOTOEDITOR_H
#define PHOTOEDITOR_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>
#include <QMessageBox>
#include <library/bmp.h>

#include "includes.h"

namespace Ui {
class PhotoEditor;
}

class PhotoEditor : public QMainWindow
{
    Q_OBJECT

public:
    explicit PhotoEditor(QWidget *parent = nullptr);
    ~PhotoEditor();

/*
 *  Load and save Image
 *  -------------------
 */
    void loadImage(const QString &dirFile);
    void saveImage(const QString &dirFile);

private slots:
    void setDisabledImageSection(const bool &_v);
    void setDisabledVideoSection(const bool &_v);

/*
 *  Read/Write BMP
 *  --------------
 */
    void on_actionAbrir_triggered();
    void on_actionGuardar_triggered();

/*
 *  Color Space Transform
 *  ---------------------
 */
    void on_actionCMY_triggered();
    void on_actionHSL_triggered();
    void on_actionHSV_triggered();
    void on_actionXYZ_triggered();
    void on_actionLMS_triggered();
    void on_actionYIQ_triggered();
    void on_actionYUV_triggered();
    void on_actionYCbCr_triggered();

    void on_actionGaussianFilter_triggered();
    void on_actionLaplaceFilter_triggered();
    void on_actionGabor0Filter_triggered();
    void on_actionGabor45Filter_triggered();
    void on_actionGabor90Filter_triggered();
    void on_actionGabor135Filter_triggered();
    
    void on_actionMagnitudeFFT_triggered();
    void on_actionPhaseFFT_triggered();

    void on_actionPerspective_triggered();

    void on_actionTrackingVideo_triggered();

private:
    Ui::PhotoEditor *ui;

    BMP   img;
    uchar *R, *G, *B;
    int   rows, cols;

    cv::VideoCapture video;

    // Graphics Pixmap
    QGraphicsPixmapItem PixOriginal;

    std::string pathTo;
};

#endif // PHOTOEDITOR_H
