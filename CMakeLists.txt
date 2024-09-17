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
find_package(PCL REQUIRED COMPONENTS common io filters octree kdtree segmentation surface features search)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(Curses REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# Add maps executables
add_executable(map src/maps_devel/map.cpp)
add_executable(map2 src/maps_devel/map2.cpp)
add_executable(map3 src/maps_devel/map3.cpp)
add_executable(map4 src/maps_devel/map4.cpp)
add_executable(cloud_test src/cloud_test.cpp)

# Add filters executables
add_executable(downsample src/filters/downsample.cpp)

# Add mesh executables
add_executable(toMesh src/toMesh/mapToMesh.cpp)
add_executable(toRviz src/toMesh/toRviz.cpp)
add_executable(SaveCloud src/toMesh/cloudSaver.cpp)
add_executable(CombineCloud src/toMesh/cloudCombine.cpp)
add_executable(MeshCloud src/toMesh/cloudMesh.cpp)


# Link libraries for maps executables
ament_target_dependencies(map rclcpp sensor_msgs pcl_conversions tf2 tf2_ros tf2_geometry_msgs cv_bridge OpenCV)
target_link_libraries(map ${PCL_LIBRARIES} Eigen3::Eigen)

ament_target_dependencies(map2 rclcpp sensor_msgs pcl_conversions tf2 tf2_ros tf2_geometry_msgs cv_bridge OpenCV)
target_link_libraries(map2 ${PCL_LIBRARIES} Eigen3::Eigen)

ament_target_dependencies(map3 rclcpp sensor_msgs pcl_conversions tf2 tf2_ros tf2_geometry_msgs cv_bridge OpenCV)
target_link_libraries(map3 ${PCL_LIBRARIES} Eigen3::Eigen)

ament_target_dependencies(map4 rclcpp sensor_msgs pcl_conversions tf2 tf2_ros tf2_geometry_msgs cv_bridge OpenCV)
target_link_libraries(map4 ${PCL_LIBRARIES} Eigen3::Eigen)

# Link libraries for filters executables
ament_target_dependencies(downsample rclcpp sensor_msgs pcl_conversions tf2 tf2_ros tf2_geometry_msgs cv_bridge OpenCV)
target_link_libraries(downsample ${PCL_LIBRARIES} Eigen3::Eigen)

# Link libraries for mesh executables
ament_target_dependencies(toMesh rclcpp sensor_msgs pcl_conversions tf2 tf2_ros tf2_geometry_msgs cv_bridge OpenCV)
target_link_libraries(toMesh ${PCL_LIBRARIES} ${CURSES_LIBRARIES} Eigen3::Eigen)

ament_target_dependencies(SaveCloud rclcpp sensor_msgs pcl_conversions tf2 tf2_ros tf2_geometry_msgs cv_bridge OpenCV)
target_link_libraries(SaveCloud ${PCL_LIBRARIES} ${CURSES_LIBRARIES} Eigen3::Eigen)

ament_target_dependencies(CombineCloud rclcpp sensor_msgs pcl_conversions tf2 tf2_ros tf2_geometry_msgs cv_bridge OpenCV)
target_link_libraries(CombineCloud ${PCL_LIBRARIES} ${CURSES_LIBRARIES} Eigen3::Eigen)

ament_target_dependencies(MeshCloud rclcpp sensor_msgs visualization_msgs pcl_conversions tf2 tf2_ros tf2_geometry_msgs cv_bridge OpenCV)
target_link_libraries(MeshCloud ${PCL_LIBRARIES} ${CURSES_LIBRARIES} Eigen3::Eigen)

ament_target_dependencies(toRviz rclcpp sensor_msgs visualization_msgs pcl_conversions tf2 tf2_ros tf2_geometry_msgs cv_bridge OpenCV)
target_link_libraries(toRviz ${PCL_LIBRARIES} ${CURSES_LIBRARIES} Eigen3::Eigen)
target_link_libraries(toRviz pcl_surface pcl_features pcl_search)

# Link libraries for cloud_test executable
ament_target_dependencies(cloud_test rclcpp sensor_msgs visualization_msgs pcl_conversions tf2 tf2_ros tf2_geometry_msgs cv_bridge OpenCV)
target_link_libraries(cloud_test ${PCL_LIBRARIES} ${CURSES_LIBRARIES} Eigen3::Eigen)



# Install targets
install(TARGETS
  map
  map2
  map3
  map4
  downsample
  toMesh
  toRviz
  SaveCloud
  CombineCloud
  MeshCloud 
  cloud_test
  DESTINATION lib/${PROJECT_NAME})

ament_package()
