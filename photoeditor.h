#ifndef PHOTOEDITOR_H
#define PHOTOEDITOR_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <library/bmp.h>

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

private:
    Ui::PhotoEditor         *ui;
    QGraphicsScene       *scene;
    QGraphicsView *graphicsView;

    BMP img;    //Estructura de tipo imágen

};

#endif // PHOTOEDITOR_H
