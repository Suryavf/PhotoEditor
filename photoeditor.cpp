#include "photoeditor.h"
#include "ui_photoeditor.h"

PhotoEditor::PhotoEditor(QWidget *parent) : QMainWindow(parent),
                                            ui(new Ui::PhotoEditor){
    ui->setupUi(this);
}

PhotoEditor::~PhotoEditor(){
    delete ui;
}
