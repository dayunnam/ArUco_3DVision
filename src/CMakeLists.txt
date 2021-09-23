cmake_minimum_required(VERSION 3.3)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} [OpenCV.cmake PATH])
project(ARuco_Calibration)
project(ARuco_Detection)
project(ARuco_3DRecon)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})
include_directories(${CERES_INCLUDE_DIRS})
 link_directories( ${OpenCV_LIB_DIR} )
 link_directories( ${CERES_INCLUDE_DIRS})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_INSTALL_PREFIX}/bin)

set(A_C_SOURCE_FILES ./ARuco_Calibration/ARuco_Calibration.cpp)
set(A_D_SOURCE_FILES ./ARuco_Detection/ARuco_Detection.cpp ./ARuco_Detection/draw_object.cpp ./ARuco_Detection/draw_object.h)
set(A_R_SOURCE_FILES ./ARuco_3DRecon/ARuco_3DRecon.cpp ./ARuco_3DRecon/bundle_adjustment.hpp ./ARuco_3DRecon/config.h ./ARuco_3DRecon/load_images.h ./ARuco_3DRecon/sfm.cpp ./ARuco_3DRecon/sfm.hpp ./ARuco_3DRecon/viewer_3d_point.h)

add_executable(ARuco_Calibration ${A_C_SOURCE_FILES})
add_executable(ARuco_Detection ${A_D_SOURCE_FILES})
add_executable(ARuco_3DRecon ${A_R_SOURCE_FILES})

target_include_directories(ARuco_Calibration PRIVATE ${OpenCV_INCLUDE_DIR} ${CERES_INCLUDE_DIRS})
target_include_directories(ARuco_Detection PRIVATE ${OpenCV_INCLUDE_DIR} ${CERES_INCLUDE_DIRS})
target_include_directories(ARuco_3DRecon PRIVATE ${OpenCV_INCLUDE_DIR} ${CERES_INCLUDE_DIRS})

target_link_libraries(ARuco_Calibration PRIVATE ${OpenCV_LIBRARIES} ${CERES_LIBRARIES})
target_link_libraries(ARuco_Detection PRIVATE ${OpenCV_LIBRARIES} ${CERES_LIBRARIES})
target_link_libraries(ARuco_3DRecon PRIVATE ${OpenCV_LIBRARIES} ${CERES_LIBRARIES})

set_target_properties(ARuco_Calibration PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set_target_properties(ARuco_Detection PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set_target_properties(ARuco_3DRecon PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
