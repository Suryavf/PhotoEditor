#ifndef PHOTOEDITOR_H
#define PHOTOEDITOR_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <library/bmp.h>

#include "library/includes.h"

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
    void on_actionAbrir_triggered();
    void on_actionGuardar_triggered();

    void on_pushButton_clicked();

    void on_actionCMY_triggered();

    void on_open_clicked();

    void on_actionHSL_triggered();

    void on_actionHSV_triggered();

    void on_actionXYZ_triggered();

    void on_actionLMS_triggered();

    void on_actionYIQ_triggered();

    void on_actionYUV_triggered();

    void on_actionYCbCr_triggered();

private:
    Ui::PhotoEditor         *ui;
    QGraphicsScene       *scene;
    QGraphicsView *graphicsView;

    BMP img;    //Estructura de tipo imágen
    uchar *R, *G, *B;

    int rows;
    int cols;
};

#endif // PHOTOEDITOR_H
