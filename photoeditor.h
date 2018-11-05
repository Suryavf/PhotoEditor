#ifndef PHOTOEDITOR_H
#define PHOTOEDITOR_H

#include <QMainWindow>

namespace Ui {
class PhotoEditor;
}

class PhotoEditor : public QMainWindow
{
    Q_OBJECT

public:
    explicit PhotoEditor(QWidget *parent = nullptr);
    ~PhotoEditor();

private:
    Ui::PhotoEditor *ui;
};

#endif // PHOTOEDITOR_H
