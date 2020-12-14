
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <set>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

// prev = query
// curr = train


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-70, bottom+30), cv::FONT_ITALIC, 1, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-70, bottom+70), cv::FONT_ITALIC, 1, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// // associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // std::vector <cv::DMatch> kptsWithROI;
    // float distMean = 0;
    // float threshold = distMean * 0.7;
    // for (auto it = kptMatches.begin(); it!=kptMatches.end(); ++it)
    // {
    //     cv::KeyPoint kp = kptsCurr.at(it->trainIdx);
        
    //     if(boundingBox.roi.contains(cv::Point(kp.pt.x, kp.pt.y))) // boundigBox의 roi에 포인트가 존재하면 kptsWithROI에 저장
    //         kptsWithROI.push_back(*it);
    // }
    // for (auto it = kptsWithROI.begin(); it!=kptsWithROI.end(); ++it)
    // {
    //     distMean = distMean + it->distance;
    // }
    // if (kptsWithROI.size()>0)
    //     distMean = distMean/kptsWithROI.size();

    // else
    //     return;
    // for  (auto it = kptsWithROI.begin(); it != kptsWithROI.end(); ++it)
    // {
    //    if (it->distance < threshold)
    //        boundingBox.kptMatches.push_back(*it);
    // }   
    // cout<<"se"<<endl;
     // Loop over all matches in the current frame

    

    for (cv::DMatch mat: kptMatches) 
    {
        if (boundingBox.roi.contains(kptsCurr[mat.trainIdx].pt)) //바운딩 박스 안에 curr 키포인트가 있는가?
        {
            boundingBox.kptMatches.push_back(mat); //있으면 바운딩박스 키포인트매치스에 저장
        }
    }
    
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
      
    vector<double> distRatios; //curr과 prev의 키포인트 사이의 거리비를 저장
    double dT = 1/frameRate;
    // cout<<kptMatches.size()<<endl; 
    
    for(auto it1 = kptMatches.begin(); it1!=kptMatches.end(); ++it1)
    {   
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

    
        for(auto it2 = kptMatches.begin(); it2 != kptMatches.end(); ++it2)
        {
            double minDist = 50.0;

            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx); 
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            double distCurr = cv::norm(kpOuterCurr.pt-kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt-kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { 

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } 
    }

   
    
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

      std::sort(distRatios.begin(),distRatios.end()); // distRatios를 정렬
    auto itr = distRatios.begin(); 
    double median;
    
    if (distRatios.size()%2 ==0)//짝수개일때 
    {

        itr = itr + (distRatios.size()/2-1);

        auto Nitr = itr+1;


        median = (*itr + *Nitr)/2;

        // cout <<distRatios.size() << endl;
        // cout<< median << endl;
    }

    if (distRatios.size()%2 !=0)
    {
        itr = itr + (distRatios.size()-1)/2;

        median = *itr;
        // for (int i = 0; i<TTC_camera.size() ; ++i)
//     {
//         cout <<"CAMERA TTC : "<< TTC_camera[i] <<endl;
//     }
// }
    }

    
    TTC = -dT / (1 - median);



    // vector <double> TTC_camera;
    // TTC_camera.push_back(TTC);

    // cout<<"CAMERA TTC : "<<endl;

}



void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    

       // auxiliary variables
    double dT = 1/frameRate; // time between two measurements in seconds
    double laneWidth = 4.0;
    vector <double> currX , prevX;


    // find closest distance to Lidar points 
    double minXPrev = 0, minXCurr = 0;

    for(auto it=lidarPointsPrev.begin(); it!=lidarPointsPrev.end(); ++it) 
    {

        if (abs(it->y) <= laneWidth / 2.0)
        { // 3D point within ego lane?
            prevX.push_back(it->x);
        }
        
    }

    for(auto it=lidarPointsCurr.begin(); it!=lidarPointsCurr.end(); ++it) 
    {
         if (abs(it->y) <= laneWidth / 2.0)
        { // 3D point within ego lane?
            currX.push_back(it->x);
        }
    }

    if (currX.size()>0)
    {
        for(auto x:currX)
        {
            minXCurr = minXCurr+x;
            
        }
        
    }
    minXCurr = minXCurr/currX.size();
    if (prevX.size()>0)
    {
        for(auto x:prevX)
        {
            minXPrev = minXPrev + x;
        }
    }
    minXPrev = minXPrev / prevX.size();

    // compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev-minXCurr);

    // cout<<"LIDAR TTC"<<endl;

    // double dT = 1 / frameRate; // time between two measurements in seconds
    // double laneWidth = 4.0;

    // double meanXPrev = 0.0, meanXCurr = 0.0;
    // int xPrevSize = 0, xCurrSize = 0;
    // for (auto itr = lidarPointsPrev.begin(); itr != lidarPointsPrev.end(); ++itr)
    // {
    //     if (abs(itr->y) > laneWidth/2.0) continue;
    //     meanXPrev += itr->x;
    //     ++xPrevSize;
    // }
    // meanXPrev /= xPrevSize;

    // for (auto itr = lidarPointsCurr.begin(); itr != lidarPointsCurr.end(); ++itr)
    // {   
    //     if (abs(itr->y) > laneWidth/2.0) continue;
    //     meanXCurr += itr->x;
    //     ++xCurrSize;
    // }
    // meanXCurr /= xCurrSize;

    // // compute TTC from both measurements
    // TTC = meanXCurr * dT / (meanXPrev - meanXCurr);

  
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // prev와 curr의 바운딩 박스를 매칭
    // bounding box 안의 keypoint들을 하나하나 매치해가며 매치된 수가 많은 짝을 매칭한다.
    int p = prevFrame.boundingBoxes.size();
    int c = currFrame.boundingBoxes.size();
    int prevKPIDX , currKPIDX;
    cv::KeyPoint prevKP, currKP;
    vector<int> prevBoxesIds, currBoxesIds;
    int count[p][c] = {};

    for (auto it1 = matches.begin(); it1 != matches.end(); ++it1)
    {

        prevBoxesIds.clear();
        currBoxesIds.clear();

        prevKPIDX = (*it1).queryIdx; //이전의 키포인트의 인덱스를 정의
        currKPIDX = (*it1).trainIdx; //현재의 키포인트 인덱스 정의

        prevKP = prevFrame.keypoints[prevKPIDX]; // 이전 프레임의 키포인트를 인덱스로 정의
        currKP = currFrame.keypoints[currKPIDX]; // 현재 프레임의 키포인트를 인덱스로 정의

        for(auto it2 = prevFrame.boundingBoxes.begin(); it2 != prevFrame.boundingBoxes.end(); ++it2 ) //바운딩 박스 안에 키포인트가 존재??
            {
                if((*it2).roi.contains(prevKP.pt)) //roi 안에 키포인트가 존재하는가?
                    prevBoxesIds.push_back((*it2).boxID); // 존재하면 BoxesIds에 박스 아이디를 넣는다
            }

        for(auto it2 = currFrame.boundingBoxes.begin(); it2 != currFrame.boundingBoxes.end(); ++it2 ) 
            {
                if((*it2).roi.contains(currKP.pt)) //roi 안에 키포인트가 존재하는가?
                    currBoxesIds.push_back((*it2).boxID); // 존재하면 BoxesIds에 박스 아이디를 넣는다
            }

        for(auto currId:currBoxesIds)
        {
            for(auto prevId:prevBoxesIds)
            {
                count[prevId][currId]++; //count 업데이트
            }
        }
    }
     for (int i = 0; i < p; i++)
    {  
         int mcount = 0;
         int id_max = 0;
         for (int j = 0; j < c; j++)
             if (count[i][j] > mcount)
             {  
                  mcount = count[i][j];
                  id_max = j;
             }
          bbBestMatches[i] = id_max;
    } 
}
// void printF (vector <double> &TTC_camera)
// {
//     for (int i = 0; i<TTC_camera.size() ; ++i)
//     {
//         cout <<"CAMERA TTC : "<< TTC_camera[i] <<endl;
//     }
// }