cmake_minimum_required(VERSION 3.5)
project(VINS_FeatureTracker_Standalone)

find_package(OpenCV REQUIRED)

add_executable(tracker
  main.cpp
  feature_tracker.cpp
)

target_include_directories(tracker PRIVATE ${OpenCV_INCLUDE_DIRS} .)

target_link_libraries(tracker PRIVATE ${OpenCV_LIBS})
