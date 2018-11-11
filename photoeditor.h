#ifndef PHOTOEDITOR_H
#define PHOTOEDITOR_H

#include <QMainWindow>
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

private:
    Ui::PhotoEditor *ui;

    BMP img;    //Estructura de tipo imágen
};

#endif // PHOTOEDITOR_H
