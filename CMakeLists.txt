cmake_minimum_required(VERSION 3.8)
project(cloud_devel)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters octree segmentation)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# Add the executable
add_executable(map src/map.cpp)
add_executable(map2 src/map2.cpp)
add_executable(map3 src/map3.cpp)
add_executable(color_test src/color_cloud_test.cpp)

# Link libraries
#map1
ament_target_dependencies(map rclcpp sensor_msgs pcl_conversions tf2 tf2_ros tf2_geometry_msgs cv_bridge OpenCV)
target_link_libraries(map ${PCL_LIBRARIES} Eigen3::Eigen)
#map2
ament_target_dependencies(map2 rclcpp sensor_msgs pcl_conversions tf2 tf2_ros tf2_geometry_msgs cv_bridge OpenCV)
target_link_libraries(map2 ${PCL_LIBRARIES} Eigen3::Eigen)
#map3
ament_target_dependencies(map3 rclcpp sensor_msgs pcl_conversions tf2 tf2_ros tf2_geometry_msgs cv_bridge OpenCV)
target_link_libraries(map3 ${PCL_LIBRARIES} Eigen3::Eigen)
#cloud test color
ament_target_dependencies(color_test rclcpp sensor_msgs pcl_conversions tf2 tf2_ros tf2_geometry_msgs cv_bridge OpenCV)
target_link_libraries(color_test ${PCL_LIBRARIES} Eigen3::Eigen)

# Install targets
install(TARGETS
  map
  map2
  map3
  color_test
  DESTINATION lib/${PROJECT_NAME})

ament_package()
