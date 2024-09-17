#include <rclcpp/rclcpp.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/io/ply_io.h>
#include <pcl/PolygonMesh.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/msg/marker.hpp>
#include <filesystem>  // For checking if file exists
#include <iostream> 

using PointT = pcl::PointXYZRGB;
using PCLcloud = pcl::PointCloud<PointT>;

class MeshGenerator : public rclcpp::Node
{
public:
    MeshGenerator() : Node("mesh_generator_node")
    {
        pub_mesh_marker_ = this->create_publisher<visualization_msgs::msg::Marker>("mesh_marker", 10);

        int user_choice = getUserChoice();

        if (user_choice == 1)
        {
            // User chose to load the existing mesh file
            if (std::filesystem::exists("mesh.ply"))
            {
                RCLCPP_INFO(this->get_logger(), "Mesh file exists, loading from mesh.ply");
                loadAndPublishMesh();
            }
            else
            {
                RCLCPP_ERROR(this->get_logger(), "Mesh file not found. Please generate a new one first.");
            }
        }
        else if (user_choice == 2)
        {
            // User chose to generate a new mesh
            generateAndPublishMesh();
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "Invalid input, terminating node.");
        }
    }

private:
    int getUserChoice()
    {
        int choice = 0;
        std::cout << "Do you want to:\n";
        std::cout << "1) Load existing mesh file (mesh.ply)\n";
        std::cout << "2) Generate a new mesh\n";
        std::cout << "Enter your choice (1 or 2): ";
        std::cin >> choice;
        return choice;
    }

    void loadAndPublishMesh()
    {
        std::string mesh_filename = "mesh.ply";
        pcl::PolygonMesh mesh;
        
        // Load the existing mesh file
        if (pcl::io::loadPLYFile(mesh_filename, mesh) == -1)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to load mesh from file %s", mesh_filename.c_str());
            return;
        }

        // Convert and publish the existing mesh as a ROS2 marker
        visualization_msgs::msg::Marker marker;
        convertMeshToMarker(mesh, marker);
        pub_mesh_marker_->publish(marker);
    }

    void generateAndPublishMesh()
    {
        std::string mesh_filename = "mesh.ply";
        RCLCPP_INFO(this->get_logger(), "Generating new mesh");

        PCLcloud::Ptr cloud(new PCLcloud);
        
        // Load the combined point cloud from file
        if (pcl::io::loadPCDFile<PointT>("combined_cloud.pcd", *cloud) == -1)
        {
            RCLCPP_ERROR(this->get_logger(), "Couldn't read file combined_cloud.pcd");
            return;
        }

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
        pcl::PolygonMesh triangles;

        // Set parameters for the triangulation
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
        gp3.reconstruct(triangles);

        // Save the mesh to a file
        pcl::io::savePLYFile(mesh_filename, triangles);
        RCLCPP_INFO(this->get_logger(), "Mesh saved to %s", mesh_filename.c_str());

        // Convert and publish the new mesh
        visualization_msgs::msg::Marker marker;
        convertMeshToMarker(triangles, marker);
        pub_mesh_marker_->publish(marker);
    }

     void convertMeshToMarker(const pcl::PolygonMesh &mesh, visualization_msgs::msg::Marker &marker)
    {
        marker.header.frame_id = "base_link";
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


    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_mesh_marker_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MeshGenerator>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
