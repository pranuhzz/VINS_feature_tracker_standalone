#include <opencv2/opencv.hpp>
#include "feature_tracker.h"
#include "tic_toc.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
  if (argc != 2) {
    cerr << "Usage: ./tracker <grayscale_image>" << endl;
    return -1;
  }

  // 1. Read image
  Mat img = imread(argv[1], IMREAD_GRAYSCALE);
  if (img.empty()) {
    cerr << "Failed to load image: " << argv[1] << endl;
    return -1;
  }

  // 2. Initialize tracker
  FeatureTracker tracker;
  tracker.readImage(img, 0.0);

  // 3. Detect / add features
  tracker.addNewFeatures();

  // 4. Retrieve points and draw
  auto pts = tracker.cur_pts;
  Mat out;
  cvtColor(img, out, COLOR_GRAY2BGR);
  for (auto &p : pts) {
    circle(out, p, 3, Scalar(0,255,0), -1);
  }

  // 5. Show result
  imshow("Detected Features", out);
  waitKey(0);
  return 0;
}
