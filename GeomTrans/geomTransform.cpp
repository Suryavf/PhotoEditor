#include "geomTransform.h"

std::vector<cv::Point2f> pts;

void sortPoints(std::vector<cv::Point2f> &outgrid ){
    float maxSum = -1.0, minSum = 999999999.99f;
    float sum;
    int idMaxSum = 0, idMinSum = 0;

    for(int i=0; i<int(pts.size()); ++i){
        sum = pts[i].x + pts[i].y;

        if(maxSum<sum){
            maxSum = sum;
            idMaxSum = i;
        }
        if(minSum>sum){
            minSum = sum;
            idMinSum = i;
        }
    }

    int minX = 99999999.999f;
    int id2 = 0, id4 = 0;
    for(int i=0; i<int(pts.size()); ++i){
        if( i!=idMaxSum && i!=idMinSum){
            sum = pts[i].x;
            if( minX>sum ){
                minX = int(sum);
                id2 = i;
            }
        }
    }

    for(int i=0; i<int(pts.size()); ++i){
        if( i!=idMaxSum && i!=idMinSum && i!=id2){
            id4 = i;
        }
    }

    outgrid.push_back( pts[idMinSum] );
    outgrid.push_back( pts[   id2  ] );
    outgrid.push_back( pts[idMaxSum] );
    outgrid.push_back( pts[   id4  ] );

    pts.assign(outgrid.begin(), outgrid.end());

    cv::Point2f d1,d2;
    d1 = pts[1] - pts[0];
    d2 = pts[2] - pts[3];
    int newCol = int( (sqrt(d1.x*d1.x + d1.y*d1.y) + sqrt(d2.x*d2.x + d2.y*d2.y))/2.0  );

    d1 = pts[3] - pts[0];
    d2 = pts[2] - pts[1];
    int newRow = int( (sqrt(d1.x*d1.x + d1.y*d1.y) + sqrt(d2.x*d2.x + d2.y*d2.y))/2.0  );

    outgrid[0] = cv::Point2f(       0 ,      0  );
    outgrid[1] = cv::Point2f(       0 ,newCol-1 );
    outgrid[2] = cv::Point2f( newRow-1,newCol-1 );
    outgrid[3] = cv::Point2f( newRow-1,      0  );
}


void CallBackFunc(int event, int x, int y, int flags, void* userdata){
     if  ( event == cv::EVENT_LBUTTONDOWN ){
          pts.push_back( cv::Point(x,y) );
     }
}

void geometricTransformation(uchar *R , uchar *G , uchar *B, uint rows, uint cols){
	// Get Mat by RGB
	std::vector<cv::Mat> array_to_merge;
    array_to_merge.push_back(cv::Mat(int(rows),int(cols),CV_8UC1,B));
    array_to_merge.push_back(cv::Mat(int(rows),int(cols),CV_8UC1,G));
    array_to_merge.push_back(cv::Mat(int(rows),int(cols),CV_8UC1,R));

    cv::Mat img, src;
    cv::merge(array_to_merge, img);

    // Windows configure
    cv::namedWindow("Geometric Transformation", 1);
    cv::setMouseCallback("Geometric Transformation", CallBackFunc, NULL);

    // Add points
    char key = 0;
    while (key != 'q' && pts.size()<4){
    	// Copy image 
    	img.copyTo(src);
    
        for(int i=0; i<int(pts.size()); ++i){
            circle( src, pts[i], 4.0, cv::Scalar( 0, 0, 255 ), 5, 8 );
    	}

        imshow("Geometric Transformation", src);
        key = char(cv::waitKey(10));
    }
    img.copyTo(src);
    for(int i=0; i<int(pts.size()); ++i)
        circle( src, pts[i], 4.0, cv::Scalar( 0, 0, 255 ), 5, 8 );
    imshow("Geometric Transformation", src);
    cv::waitKey(10);

    std::vector<cv::Point2f> outgrid;
    sortPoints( outgrid );

    // Transform
    cv::Point2f * inputQuad = &pts    [0];
    cv::Point2f *outputQuad = &outgrid[0];

    // Get the Perspective Transform Matrix i.e. lambda
    cv::Mat lambda( 2, 4, CV_32FC1 );
    lambda = getPerspectiveTransform( inputQuad, outputQuad );

    cv::Mat out;
    warpPerspective(img,out,lambda, cv::Size(outputQuad[2].x+1,outputQuad[2].y+1) );

    imshow("Out", out);
    cv::waitKey(0);
}
