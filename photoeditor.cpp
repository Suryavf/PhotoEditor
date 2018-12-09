#include "photoeditor.h"
#include "ui_photoeditor.h"
#include <QtCore>
#include <QtGui>
#include <QFileDialog>
#include <library/bmp.h>

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
