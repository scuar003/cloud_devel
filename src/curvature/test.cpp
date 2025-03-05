#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/PolygonMesh.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using PointT = pcl::PointXYZRGB;
using PCLcloud = pcl::PointCloud<PointT>;

class MeshGenerator : public rclcpp::Node
{
public:
    MeshGenerator() : Node("mesh_generator_node"), tf_buffer(this->get_clock()), tf_listener(tf_buffer)
    {
        sub_pointcloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/camera/depth/color/points", 10, 
            std::bind(&MeshGenerator::cloudCallback, this, std::placeholders::_1));

        pub_mesh_marker_ = this->create_publisher<visualization_msgs::msg::Marker>("mesh_marker", 10);
    }

private:
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Convert ROS2 message to PCL PointCloud
        PCLcloud::Ptr cloud(new PCLcloud);
        pcl::fromROSMsg(*msg, *cloud);

        // Transform the point cloud to the base_link frame
        if (!transformPointCloudToBaseLink(cloud, msg->header.frame_id))
            return;

        // Generate and publish the mesh
        pcl::PolygonMesh mesh;
        if (generateMesh(cloud, mesh))
        {
            visualization_msgs::msg::Marker marker;
            convertMeshToMarker(mesh, marker);
            pub_mesh_marker_->publish(marker);
        }
    }

    bool transformPointCloudToBaseLink(PCLcloud::Ptr &cloud, const std::string &frame_id)
    {
        geometry_msgs::msg::TransformStamped transform_stamped;
        try
        {
            transform_stamped = tf_buffer.lookupTransform("camera_link", frame_id, tf2::TimePointZero);
        }
        catch (tf2::TransformException &ex)
        {
            RCLCPP_WARN(this->get_logger(), "Could not transform %s to base_link: %s", frame_id.c_str(), ex.what());
            return false;
        }

        Eigen::Affine3d transform = Eigen::Affine3d::Identity();
        transform.translation() << transform_stamped.transform.translation.x,
            transform_stamped.transform.translation.y,
            transform_stamped.transform.translation.z;
        Eigen::Quaterniond q(transform_stamped.transform.rotation.w,
                            transform_stamped.transform.rotation.x,
                            transform_stamped.transform.rotation.y,
                            transform_stamped.transform.rotation.z);
        transform.rotate(q);
        pcl::transformPointCloud(*cloud, *cloud, transform);
        return true;
    }

    bool generateMesh(const PCLcloud::Ptr &cloud, pcl::PolygonMesh &mesh)
    {
        // Estimate normals
        pcl::NormalEstimation<PointT, pcl::Normal> ne;
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
        tree->setInputCloud(cloud);
        ne.setInputCloud(cloud);
        ne.setSearchMethod(tree);
        ne.setKSearch(20);
        ne.compute(*normals);

        // Concatenate the point cloud with its normals
        pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
        pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);

        // Create a KD-Tree
        pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
        tree2->setInputCloud(cloud_with_normals);

        // Initialize objects for triangulation
        pcl::GreedyProjectionTriangulation<pcl::PointXYZRGBNormal> gp3;
        gp3.setSearchRadius(0.025);
        gp3.setMu(2.5);
        gp3.setMaximumNearestNeighbors(100);
        gp3.setMaximumSurfaceAngle(M_PI / 4); // 45 degrees
        gp3.setMinimumAngle(M_PI / 18); // 10 degrees
        gp3.setMaximumAngle(2 * M_PI / 3); // 120 degrees
        gp3.setNormalConsistency(false);

        // Perform the triangulation
        gp3.setInputCloud(cloud_with_normals);
        gp3.setSearchMethod(tree2);
        gp3.reconstruct(mesh);

        return !mesh.polygons.empty();
    }

    void convertMeshToMarker(const pcl::PolygonMesh &mesh, visualization_msgs::msg::Marker &marker)
    {
        marker.header.frame_id = "map";
        marker.header.stamp = this->get_clock()->now();
        marker.ns = "mesh";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;

        pcl::PointCloud<pcl::PointXYZRGBNormal> cloud;
        pcl::fromPCLPointCloud2(mesh.cloud, cloud);

        for (const auto &polygon : mesh.polygons)
        {
            if (polygon.vertices.size() == 3)
            {
                for (const auto &vertex : polygon.vertices)
                {
                    geometry_msgs::msg::Point p;
                    p.x = cloud.points[vertex].x;
                    p.y = cloud.points[vertex].y;
                    p.z = cloud.points[vertex].z;
                    marker.points.push_back(p);

                    std_msgs::msg::ColorRGBA color;
                    color.r = cloud.points[vertex].r / 255.0;
                    color.g = cloud.points[vertex].g / 255.0;
                    color.b = cloud.points[vertex].b / 255.0;
                    color.a = 1.0;
                    marker.colors.push_back(color);
                }
            }
        }

        marker.scale.x = 1.0;
        marker.scale.y = 1.0;
        marker.scale.z = 1.0;
        marker.color.a = 1.0;
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pointcloud_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_mesh_marker_;
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MeshGenerator>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
