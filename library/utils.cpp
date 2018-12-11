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
        cv::Size newsize(big.cols/coff,big.rows/coff);
        cv::resize(big, big, newsize, 0, 0, CV_INTER_LINEAR);
    }

    // Show
    cv::namedWindow( name );
    cv::imshow( name, big );
}
