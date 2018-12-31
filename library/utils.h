#ifndef UTILS_H
#define UTILS_H

#include "../includes.h"
#include <QFileInfo>

void showColorModel(uchar* &C1,uchar* &C2,uchar* &C3,int rows,int cols,cv::String name);
void abrir_imagen(uchar* &R, uchar* &G, uchar* &B,int &rows, int &cols, const std::string &pathTo);

bool check_BMP_format(const std::string &pathTo);
bool checkImageformat(const std::string &pathTo);
bool checkVideoformat(const std::string &pathTo);

#endif // UTILS_H
