#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/marching_cubes_hoppe.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>

using PointT = pcl::PointXYZRGB;

class PointCloudToMeshNode : public rclcpp::Node
{
public:
  PointCloudToMeshNode() : Node("pointcloud_to_mesh_node")
  {
    pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/camera/depth/color/points", 10, std::bind(&PointCloudToMeshNode::pointcloud_callback, this, std::placeholders::_1));
    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("mesh_marker", 10);
  }

private:
  void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*msg, *cloud);

    // Downsample the point cloud using a VoxelGrid filter
    pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
    pcl::VoxelGrid<PointT> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(0.01f, 0.01f, 0.01f);
    sor.filter(*cloud_filtered);

    // Create the normal estimation class, and pass the input dataset to it
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud_filtered);
    ne.setInputCloud(cloud_filtered);
    ne.setSearchMethod(tree);
    ne.setKSearch(20);
    ne.compute(*cloud_normals);

    // Combine points and normals
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::concatenateFields(*cloud_filtered, *cloud_normals, *cloud_with_normals);

    // Create search tree
    pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
    tree2->setInputCloud(cloud_with_normals);

    // Create the Marching Cubes object
    pcl::MarchingCubesHoppe<pcl::PointXYZRGBNormal> mc;
    pcl::PolygonMesh triangles;

    // Set input cloud and reconstruct
    mc.setInputCloud(cloud_with_normals);
    mc.reconstruct(triangles);

    // Convert to visualization_msgs/Marker message and publish
    visualization_msgs::msg::Marker marker_msg;
    marker_msg.header.stamp = this->now();
    marker_msg.header.frame_id = msg->header.frame_id;
    marker_msg.ns = "mesh";
    marker_msg.id = 0;
    marker_msg.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
    marker_msg.action = visualization_msgs::msg::Marker::ADD;
    marker_msg.pose.orientation.w = 1.0;
    marker_msg.scale.x = 1.0;
    marker_msg.scale.y = 1.0;
    marker_msg.scale.z = 1.0;
    marker_msg.color.a = 1.0;
    marker_msg.color.r = 0.0;
    marker_msg.color.g = 1.0;
    marker_msg.color.b = 0.0;

    // Convert the vertices and triangles
    pcl::PointCloud<pcl::PointXYZ> mesh_vertices;
    pcl::fromPCLPointCloud2(triangles.cloud, mesh_vertices);
    for (const auto& polygon : triangles.polygons) {
      for (const auto& vertex : polygon.vertices) {
        const auto& point = mesh_vertices.points[vertex];
        geometry_msgs::msg::Point p;
        p.x = point.x;
        p.y = point.y;
        p.z = point.z;
        marker_msg.points.push_back(p);
      }
    }

    marker_pub_->publish(marker_msg);
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PointCloudToMeshNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
