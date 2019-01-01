#include "utils.h"

void showColorModel(uchar* &C1,uchar* &C2,uchar* &C3,int rows,int cols,cv::String name){



    cv::Mat big;
    cv::Mat matArray[] = { cv::Mat(rows,cols,CV_8UC1,C1),
                           cv::Mat(rows,cols,CV_8UC1,C2),
                           cv::Mat(rows,cols,CV_8UC1,C3)};
    cv::hconcat(matArray,3,big);

    // Resize
    float coeffCol, coeffRow,coff;
    coeffCol = float(big.cols)/1910.0f;
    coeffRow = float(big.rows)/1080.0f;

    if(coeffCol>1.0f || coeffRow>1.0f){
        if(coeffCol>coeffRow) coff = coeffCol;
        else                  coff = coeffRow;
        cv::Size newsize(int(big.cols/coff),int(big.rows/coff));
        cv::resize(big, big, newsize, 0, 0, CV_INTER_LINEAR);
    }

    // Show
    cv::namedWindow( name );
    cv::imshow( name, big );
}

void abrir_imagen(uchar* &R, uchar* &G, uchar* &B,
                  int &rows, int &cols,
                  const std::string &pathTo){
    // Getting data
    cv::Mat img = cv::imread(pathTo);
    rows = img.rows;
    cols = img.cols;

    R = new uchar[rows*cols];
    G = new uchar[rows*cols];
    B = new uchar[rows*cols];

#   pragma omp parallel for collapse(2) num_threads(4)
    for(int i=0; i<rows; i++) for(int j=0; j<cols; j++){
        int id = j + i*cols;
        B[id] = img.at<cv::Vec3b>(i,j)[0];
        G[id] = img.at<cv::Vec3b>(i,j)[1];
        R[id] = img.at<cv::Vec3b>(i,j)[2];
    }
}


bool check_BMP_format(const std::string &pathTo){
    QFileInfo fi(QString::fromStdString(pathTo));
    QString ext = fi.suffix();

    if ( ext == "bmp" ) return  true;
    else                return false;
}


bool checkImageformat(const std::string &pathTo){
    QFileInfo fi(QString::fromStdString(pathTo));
    QString ext = fi.suffix();

         if ( ext ==  "bmp" ) return  true;
    else if ( ext ==  "png" ) return  true;
    else if ( ext ==  "jpg" ) return  true;
    else if ( ext ==  "gif" ) return  true;
    else if ( ext ==  "tif" ) return  true;
    else if ( ext == "jpeg" ) return  true;
    else                      return false;
}


bool checkVideoformat(const std::string &pathTo){
    QFileInfo fi(QString::fromStdString(pathTo));
    QString ext = fi.suffix();

         if ( ext ==  "mp4" ) return  true;
    else if ( ext ==  "avi" ) return  true;
    else if ( ext ==  "mpg" ) return  true;
    else if ( ext == "mpeg" ) return  true;
    else                      return false;
}

