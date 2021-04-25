#include "geometry.h"
#include <random>

class Octree_node {
private:
  int m_id;
  Bounding_box m_bounding_box;
  int m_begin, m_end;

public:
  // Constructor.
  __host__ __device__ Octree_node() : m_id(0), m_begin(0), m_end(0) {}

  // The ID of a node at its level.
  __host__ __device__ int id() const { return m_id; }

  // The ID of a node at its level.
  __host__ __device__ void set_id(int new_id) { m_id = new_id; }

  // The bounding box.
  __host__ __device__ __forceinline__ const Bounding_box &bounding_box() const {
    return m_bounding_box;
  }

  // Set the bounding box.
  __host__ __device__ __forceinline__ void
  set_bounding_box(float min_x, float min_y, float min_z, float max_x,
                   float max_y, float max_z) {
    m_bounding_box.set(min_x, min_y, min_z, max_x, max_y, max_z);
  }

  // The number of points in the tree.
  __host__ __device__ __forceinline__ int num_points() const {
    return m_end - m_begin;
  }

  // The range of points in the tree.
  __host__ __device__ __forceinline__ int points_begin() const {
    return m_begin;
  }

  __host__ __device__ __forceinline__ int points_end() const { return m_end; }

  // Define the range for that node.
  __host__ __device__ __forceinline__ void set_range(int begin, int end) {
    m_begin = begin;
    m_end = end;
  }
};

struct Parameters {
  // Choose the right set of points to use as in/out.
  int point_selector;
  // The number of nodes at a given level (2^k for level k).
  int num_nodes_at_this_level;
  // The recursion depth.
  int depth;
  // The max value for depth.
  const int max_depth;
  // The minimum number of points in a node to stop recursion.
  const int min_points_per_node;

  // Constructor set to default values.
  __host__ __device__ Parameters(int max_depth, int min_points_per_node)
      : point_selector(0), num_nodes_at_this_level(1), depth(0),
        max_depth(max_depth), min_points_per_node(min_points_per_node) {}

  // Copy constructor. Changes the values for next iteration.
  __host__ __device__ Parameters(const Parameters &params, bool)
      : point_selector((params.point_selector + 1) % 2),
        num_nodes_at_this_level(4 * params.num_nodes_at_this_level),
        depth(params.depth + 1), max_depth(params.max_depth),
        min_points_per_node(params.min_points_per_node) {}
};

bool check_octree(const Octree_node *nodes, size_t idx, size_t num_pts,
                  Points *pts, Parameters params) {
  const Octree_node &node = nodes[idx];
  int num_points = node.num_points();

  if (params.depth == params.max_depth ||
      num_points <= params.min_points_per_node) {
    int num_points_in_children = 0;

    num_points_in_children +=
        nodes[params.num_nodes_at_this_level + 8 * idx + 0].num_points();
    num_points_in_children +=
        nodes[params.num_nodes_at_this_level + 8 * idx + 1].num_points();
    num_points_in_children +=
        nodes[params.num_nodes_at_this_level + 8 * idx + 2].num_points();
    num_points_in_children +=
        nodes[params.num_nodes_at_this_level + 8 * idx + 3].num_points();
    num_points_in_children +=
        nodes[params.num_nodes_at_this_level + 8 * idx + 4].num_points();
    num_points_in_children +=
        nodes[params.num_nodes_at_this_level + 8 * idx + 5].num_points();
    num_points_in_children +=
        nodes[params.num_nodes_at_this_level + 8 * idx + 6].num_points();
    num_points_in_children +=
        nodes[params.num_nodes_at_this_level + 8 * idx + 7].num_points();

    if (num_points_in_children != node.num_points())
      return false;

    return check_quadtree(&nodes[params.num_nodes_at_this_level], 8 * idx + 0,
                          num_pts, pts, Parameters(params, true)) &&
           check_quadtree(&nodes[params.num_nodes_at_this_level], 8 * idx + 1,
                          num_pts, pts, Parameters(params, true)) &&
           check_quadtree(&nodes[params.num_nodes_at_this_level], 8 * idx + 2,
                          num_pts, pts, Parameters(params, true)) &&
           check_quadtree(&nodes[params.num_nodes_at_this_level], 8 * idx + 3,
                          num_pts, pts, Parameters(params, true)) &&
           check_quadtree(&nodes[params.num_nodes_at_this_level], 8 * idx + 4,
                          num_pts, pts, Parameters(params, true)) &&
           check_quadtree(&nodes[params.num_nodes_at_this_level], 8 * idx + 5,
                          num_pts, pts, Parameters(params, true)) &&
           check_quadtree(&nodes[params.num_nodes_at_this_level], 8 * idx + 6,
                          num_pts, pts, Parameters(params, true)) &&
           check_quadtree(&nodes[params.num_nodes_at_this_level], 8 * idx + 7,
                          num_pts, pts, Parameters(params, true));
  }

  const Bounding_box &bbox = node.bounding_box();

  for (int it = node.points_begin(); it < node.points_end(); ++it) {
    if (it >= num_pts)
      return false;

    float3 p = pts->get_point(it);

    if (!bbox.contains(p))
      return false;
  }

  return true;
}
