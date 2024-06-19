cmake_minimum_required(VERSION 3.10)

project(MainProject LANGUAGES CXX)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")

include_directories(
    /opt/homebrew/Cellar/opencv/4.9.0_9/include/opencv4/
    ./
)
link_directories(/opt/homebrew/Cellar/opencv/4.9.0_9/lib/)

file(GLOB SOURCES
    "*.cpp"
    "lights/*.cpp"
    "objects/*.cpp"
    "textures/*.cpp"
)

add_executable(main ${SOURCES})

target_link_libraries(main PUBLIC ${OpenCV_LIBS})
