#include "matching.h"

std::vector<cv::Point> vertex;

void clickevent(int event, int x, int y, int flags, void* userdata){
     if  ( event == cv::EVENT_LBUTTONDOWN ){
          vertex.push_back( cv::Point(x,y) );
     }
}

void minRectagle(std::vector<cv::Point> &pts, cv::Rect &rect){

    int xmin=99999999,ymin=99999999,xmax=-1,ymax=-1;
    int x,y;
    for(size_t i = 0; i<pts.size();++i){
        x = int(pts[i].x);
        y = int(pts[i].y);

        if(xmin > x) xmin = x;
        if(ymin > y) ymin = y;
        if(xmax < x) xmax = x;
        if(ymax < y) ymax = y;
    }

    rect = cv::Rect( cv::Point(xmin,ymin),
                     cv::Point(xmax,ymax));
}

void minmax( cv::Mat &result ,cv::Point &position ){
    double minVal, maxVal;
    cv::Point  minLoc, maxLoc, matchLoc;

    minMaxLoc( result, &minVal, &maxVal, &position, &maxLoc, cv::Mat() );
    //position = minLoc;
}


void matching(cv::VideoCapture &capture){
/*
 *  Open Video
 *  ----------
 */
    cv::Mat img, src, gray;
    cv::String nameWindow = "Video";

    // Windows configure
    cv::namedWindow(nameWindow, 1);
    cv::setMouseCallback(nameWindow, clickevent, NULL);

/*
 *  Capture First Frame
 *  -------------------
 */
    capture >> img;
    cvtColor(img, gray, CV_BGR2GRAY);
    GaussianBlur( gray, gray, cv::Size(7,7), 3.0 );

    while (vertex.size()<4){
        img.copyTo(src);

        for(size_t i=0; i<vertex.size(); ++i)
            circle( src, vertex[i], 3.0, cv::Scalar( 0, 0, 255 ), 4, 8 );

        imshow(nameWindow, src);
        cv::waitKey(10);
    }

/*
 *  Define template
 *  ---------------
 */
    cv::Rect roi;
    minRectagle(vertex,roi);

    cv::rectangle(img, roi, cv::Scalar(0, 0, 255),4);
    imshow(nameWindow, img);
    cv::waitKey(10);

    cv::Mat templ = gray( roi );

/*
 *  Video Loop
 *  ----------
 */
    cv::Mat result;
    cv::Point post;
    char key = 0;
    while (key != 'q'){
        // Get frame
        capture >> img;

        cvtColor(img, gray, CV_BGR2GRAY);
        GaussianBlur( gray, gray, cv::Size(7,7), 3.0 );

        matchTemplate(gray,templ,result, CV_TM_CCORR_NORMED);
        normalize( result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );

        double a,b;
        cv::Point  c;
        cv::minMaxLoc( result, &a, &b, &post, &c);

        // Show image
        cv::circle( img, post, 3.0, cv::Scalar( 255, 0, 0 ), 4, 8 );
        cv::rectangle( img, post, cv::Point( post.x + templ.cols , post.y + templ.rows ), cv::Scalar( 0, 0, 255 ), 5 );

        imshow(nameWindow, img);
        key = char(cv::waitKey(10));
    }

}
