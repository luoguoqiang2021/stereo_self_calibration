cmake_minimum_required(VERSION 3.0.0)
project(selfcalib VERSION 1.0.0)

add_definitions(-std=c++11)
add_definitions(-w)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE RELEASE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

include_directories(./inc)
link_directories (${PROJECT_SOURCE_DIR}/lib)

ADD_EXECUTABLE(${PROJECT_NAME} demo/main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} libselfcalib.so)
