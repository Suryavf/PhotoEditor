#ifndef PHOTOEDITOR_H
#define PHOTOEDITOR_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <QGraphicsView>
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
    void on_pushButton_clicked();

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

private:
    Ui::PhotoEditor         *ui;
    QGraphicsScene       *scene;
    QGraphicsView *graphicsView;

    BMP   img;
    uchar *R, *G, *B;

    int rows;
    int cols;

    std::string pathTo;
};

#endif // PHOTOEDITOR_H
