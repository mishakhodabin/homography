CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

FIND_PACKAGE(OpenCV REQUIRED)

PROJECT(TP5)

ADD_EXECUTABLE(homographie homographie.cpp image.cpp image.h)
TARGET_LINK_LIBRARIES(homographie ${OpenCV_LIBS})
