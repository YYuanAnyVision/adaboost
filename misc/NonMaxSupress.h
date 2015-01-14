#ifndef NONMAXSUPRESS_H
#define NONMAXSUPRESS_H

#include "opencv2/opencv.hpp"

#define NMS_MAX     1
#define NMS_MAXG    2

/*
 * Non-Maximum Supress
 * INPUT
 *  boxes               - bounding box
 *  scores              - score for each box
 *  overlap_treshold    - when multiple box with overlap ratio higer than
 *                      overlap_threshold, keep the one with highest score
 *  tpye                - default is NMS_MAX, NMS_MAXG is greedy verison of
 *                      NMS_MAX, which supressed box doesn't supress over box.
 *
 * OUTPU
 *  boxes               - supress result
 */
void NonMaxSupress(std::vector<cv::Rect> &boxes, std::vector<double> &scores,
                   double overlap_threshold = 0.3, int type=NMS_MAX);

#endif // NONMAXSUPRESS_H
