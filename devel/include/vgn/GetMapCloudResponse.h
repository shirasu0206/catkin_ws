// Generated by gencpp from file vgn/GetMapCloudResponse.msg
// DO NOT EDIT!


#ifndef VGN_MESSAGE_GETMAPCLOUDRESPONSE_H
#define VGN_MESSAGE_GETMAPCLOUDRESPONSE_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <sensor_msgs/PointCloud2.h>

namespace vgn
{
template <class ContainerAllocator>
struct GetMapCloudResponse_
{
  typedef GetMapCloudResponse_<ContainerAllocator> Type;

  GetMapCloudResponse_()
    : voxel_size(0.0)
    , map_cloud()  {
    }
  GetMapCloudResponse_(const ContainerAllocator& _alloc)
    : voxel_size(0.0)
    , map_cloud(_alloc)  {
  (void)_alloc;
    }



   typedef double _voxel_size_type;
  _voxel_size_type voxel_size;

   typedef  ::sensor_msgs::PointCloud2_<ContainerAllocator>  _map_cloud_type;
  _map_cloud_type map_cloud;





  typedef boost::shared_ptr< ::vgn::GetMapCloudResponse_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::vgn::GetMapCloudResponse_<ContainerAllocator> const> ConstPtr;

}; // struct GetMapCloudResponse_

typedef ::vgn::GetMapCloudResponse_<std::allocator<void> > GetMapCloudResponse;

typedef boost::shared_ptr< ::vgn::GetMapCloudResponse > GetMapCloudResponsePtr;
typedef boost::shared_ptr< ::vgn::GetMapCloudResponse const> GetMapCloudResponseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::vgn::GetMapCloudResponse_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::vgn::GetMapCloudResponse_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::vgn::GetMapCloudResponse_<ContainerAllocator1> & lhs, const ::vgn::GetMapCloudResponse_<ContainerAllocator2> & rhs)
{
  return lhs.voxel_size == rhs.voxel_size &&
    lhs.map_cloud == rhs.map_cloud;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::vgn::GetMapCloudResponse_<ContainerAllocator1> & lhs, const ::vgn::GetMapCloudResponse_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace vgn

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::vgn::GetMapCloudResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::vgn::GetMapCloudResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::vgn::GetMapCloudResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::vgn::GetMapCloudResponse_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::vgn::GetMapCloudResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::vgn::GetMapCloudResponse_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::vgn::GetMapCloudResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "40b2d5c19b1ba0f3eae539bbfbf252ac";
  }

  static const char* value(const ::vgn::GetMapCloudResponse_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x40b2d5c19b1ba0f3ULL;
  static const uint64_t static_value2 = 0xeae539bbfbf252acULL;
};

template<class ContainerAllocator>
struct DataType< ::vgn::GetMapCloudResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "vgn/GetMapCloudResponse";
  }

  static const char* value(const ::vgn::GetMapCloudResponse_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::vgn::GetMapCloudResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "float64 voxel_size\n"
"sensor_msgs/PointCloud2 map_cloud\n"
"\n"
"\n"
"================================================================================\n"
"MSG: sensor_msgs/PointCloud2\n"
"# This message holds a collection of N-dimensional points, which may\n"
"# contain additional information such as normals, intensity, etc. The\n"
"# point data is stored as a binary blob, its layout described by the\n"
"# contents of the \"fields\" array.\n"
"\n"
"# The point cloud data may be organized 2d (image-like) or 1d\n"
"# (unordered). Point clouds organized as 2d images may be produced by\n"
"# camera depth sensors such as stereo or time-of-flight.\n"
"\n"
"# Time of sensor data acquisition, and the coordinate frame ID (for 3d\n"
"# points).\n"
"Header header\n"
"\n"
"# 2D structure of the point cloud. If the cloud is unordered, height is\n"
"# 1 and width is the length of the point cloud.\n"
"uint32 height\n"
"uint32 width\n"
"\n"
"# Describes the channels and their layout in the binary data blob.\n"
"PointField[] fields\n"
"\n"
"bool    is_bigendian # Is this data bigendian?\n"
"uint32  point_step   # Length of a point in bytes\n"
"uint32  row_step     # Length of a row in bytes\n"
"uint8[] data         # Actual point data, size is (row_step*height)\n"
"\n"
"bool is_dense        # True if there are no invalid points\n"
"\n"
"================================================================================\n"
"MSG: std_msgs/Header\n"
"# Standard metadata for higher-level stamped data types.\n"
"# This is generally used to communicate timestamped data \n"
"# in a particular coordinate frame.\n"
"# \n"
"# sequence ID: consecutively increasing ID \n"
"uint32 seq\n"
"#Two-integer timestamp that is expressed as:\n"
"# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n"
"# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n"
"# time-handling sugar is provided by the client library\n"
"time stamp\n"
"#Frame this data is associated with\n"
"string frame_id\n"
"\n"
"================================================================================\n"
"MSG: sensor_msgs/PointField\n"
"# This message holds the description of one point entry in the\n"
"# PointCloud2 message format.\n"
"uint8 INT8    = 1\n"
"uint8 UINT8   = 2\n"
"uint8 INT16   = 3\n"
"uint8 UINT16  = 4\n"
"uint8 INT32   = 5\n"
"uint8 UINT32  = 6\n"
"uint8 FLOAT32 = 7\n"
"uint8 FLOAT64 = 8\n"
"\n"
"string name      # Name of field\n"
"uint32 offset    # Offset from start of point struct\n"
"uint8  datatype  # Datatype enumeration, see above\n"
"uint32 count     # How many elements in the field\n"
;
  }

  static const char* value(const ::vgn::GetMapCloudResponse_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::vgn::GetMapCloudResponse_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.voxel_size);
      stream.next(m.map_cloud);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct GetMapCloudResponse_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::vgn::GetMapCloudResponse_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::vgn::GetMapCloudResponse_<ContainerAllocator>& v)
  {
    s << indent << "voxel_size: ";
    Printer<double>::stream(s, indent + "  ", v.voxel_size);
    s << indent << "map_cloud: ";
    s << std::endl;
    Printer< ::sensor_msgs::PointCloud2_<ContainerAllocator> >::stream(s, indent + "  ", v.map_cloud);
  }
};

} // namespace message_operations
} // namespace ros

#endif // VGN_MESSAGE_GETMAPCLOUDRESPONSE_H
