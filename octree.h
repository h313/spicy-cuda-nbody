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
                  Particles *pts, Parameters params) {
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

    return check_octree(&nodes[params.num_nodes_at_this_level], 8 * idx + 0,
                        num_pts, pts, Parameters(params, true)) &&
           check_octree(&nodes[params.num_nodes_at_this_level], 8 * idx + 1,
                        num_pts, pts, Parameters(params, true)) &&
           check_octree(&nodes[params.num_nodes_at_this_level], 8 * idx + 2,
                        num_pts, pts, Parameters(params, true)) &&
           check_octree(&nodes[params.num_nodes_at_this_level], 8 * idx + 3,
                        num_pts, pts, Parameters(params, true)) &&
           check_octree(&nodes[params.num_nodes_at_this_level], 8 * idx + 4,
                        num_pts, pts, Parameters(params, true)) &&
           check_octree(&nodes[params.num_nodes_at_this_level], 8 * idx + 5,
                        num_pts, pts, Parameters(params, true)) &&
           check_octree(&nodes[params.num_nodes_at_this_level], 8 * idx + 6,
                        num_pts, pts, Parameters(params, true)) &&
           check_octree(&nodes[params.num_nodes_at_this_level], 8 * idx + 7,
                        num_pts, pts, Parameters(params, true));
  }

  const Bounding_box &bbox = node.bounding_box();

  for (int it = node.points_begin(); it < node.points_end(); ++it) {
    if (it >= num_pts)
      return false;

    float3 p = pts->get_location(it);

    if (!bbox.contains(p))
      return false;
  }

  return true;
}

template <int NUM_THREADS_PER_BLOCK>
__global__ void build_octree_kernel(Octree_node *nodes, Particles *points,
                                    Parameters params) {
  // The number of warps in a block.
  const int NUM_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / warpSize;

  // Shared memory to store the number of points.
  extern __shared__ int smem[];

  // s_num_pts[4][NUM_WARPS_PER_BLOCK];
  // Addresses of shared memory.
  volatile int *s_num_pts[8];

  for (int i = 0; i < 8; ++i)
    s_num_pts[i] = (volatile int *)&smem[i * NUM_WARPS_PER_BLOCK];

  // Compute the coordinates of the threads in the block.
  const int warp_id = threadIdx.x / warpSize;
  const int lane_id = threadIdx.x % warpSize;

  // TODO: Mask for compaction.
  int lane_mask_lt = (1 << lane_id) - 1;

  // The current node.
  Octree_node &node = nodes[blockIdx.x];
  node.set_id(node.id() + blockIdx.x);

  // The number of points in the node.
  int num_points = node.num_points();

  //
  // 1- Check the number of points and its depth.
  //

  // Stop the recursion here. Make sure points[0] contains all the points.
  if (params.depth >= params.max_depth ||
      num_points <= params.min_points_per_node) {
    if (params.point_selector == 1) {
      int it = node.points_begin(), end = node.points_end();

      for (it += threadIdx.x; it < end; it += NUM_THREADS_PER_BLOCK)
        if (it < end)
          points[0].set_location(it, points[1].get_location(it));
    }

    return;
  }

  // Compute the center of the bounding box of the points.
  const Bounding_box &bbox = node.bounding_box();
  float3 center;
  bbox.compute_center(center);

  // Find how many points to give to each warp.
  int num_points_per_warp = max(
      warpSize, (num_points + NUM_WARPS_PER_BLOCK - 1) / NUM_WARPS_PER_BLOCK);

  // Each warp of threads will compute the number of points to move to each
  // quadrant.
  int range_begin = node.points_begin() + warp_id * num_points_per_warp;
  int range_end = min(range_begin + num_points_per_warp, node.points_end());

  //
  // 2- Count the number of points in each child.
  //

  // Reset the counts of points per child.
  if (lane_id == 0) {
    s_num_pts[0][warp_id] = 0;
    s_num_pts[1][warp_id] = 0;
    s_num_pts[2][warp_id] = 0;
    s_num_pts[3][warp_id] = 0;
    s_num_pts[4][warp_id] = 0;
    s_num_pts[5][warp_id] = 0;
    s_num_pts[6][warp_id] = 0;
    s_num_pts[7][warp_id] = 0;
  }

  // Input points.
  const Particles &in_points = points[params.point_selector];

  // Compute the number of points.
  for (int range_it = range_begin + lane_id; __any(range_it < range_end);
       range_it += warpSize) {
    // Is it still an active thread?
    bool is_active = range_it < range_end;

    // Load the coordinates of the point.
    float3 p = is_active ? in_points.get_location(range_it)
                         : make_float3(0.0f, 0.0f, 0.0f);

    int num_pts = 0;

    // Count top-left points in front
    num_pts = __popc(__ballot(is_active && p.x < center.x && p.y >= center.y &&
                              p.z >= center.z));

    if (num_pts > 0 && lane_id == 0)
      s_num_pts[0][warp_id] += num_pts;

    // Count top-right points in front
    num_pts = __popc(__ballot(is_active && p.x >= center.x && p.y >= center.y &&
                              p.z >= center.z));

    if (num_pts > 0 && lane_id == 0)
      s_num_pts[1][warp_id] += num_pts;

    // Count bottom-left points in front
    num_pts = __popc(__ballot(is_active && p.x < center.x && p.y < center.y &&
                              p.z >= center.z));

    if (num_pts > 0 && lane_id == 0)
      s_num_pts[2][warp_id] += num_pts;

    // Count bottom-right points in front
    num_pts = __popc(__ballot(is_active && p.x >= center.x && p.y < center.y &&
                              p.z >= center.z));

    if (num_pts > 0 && lane_id == 0)
      s_num_pts[3][warp_id] += num_pts;

    // Count bottom-right points in back
    num_pts = __popc(__ballot(is_active && p.x < center.x && p.y >= center.y &&
                              p.z < center.z));

    if (num_pts > 0 && lane_id == 0)
      s_num_pts[4][warp_id] += num_pts;

    // Count top-right points in back
    num_pts = __popc(__ballot(is_active && p.x >= center.x && p.y >= center.y &&
                              p.z < center.z));

    if (num_pts > 0 && lane_id == 0)
      s_num_pts[5][warp_id] += num_pts;

    // Count bottom-left points in back
    num_pts = __popc(__ballot(is_active && p.x < center.x && p.y < center.y &&
                              p.z < center.z));

    if (num_pts > 0 && lane_id == 0)
      s_num_pts[6][warp_id] += num_pts;

    // Count bottom-right points in back
    num_pts = __popc(__ballot(is_active && p.x >= center.x && p.y < center.y &&
                              p.z < center.z));

    if (num_pts > 0 && lane_id == 0)
      s_num_pts[7][warp_id] += num_pts;
  }

  // Make sure warps have finished counting.
  __syncthreads();

  //
  // 3- Scan the warps' results to know the "global" numbers.
  // TODO: fix

  // First 4 warps scan the numbers of points per child (inclusive scan).
  if (warp_id < 8) {
    int num_pts =
        lane_id < NUM_WARPS_PER_BLOCK ? s_num_pts[warp_id][lane_id] : 0;
#pragma unroll

    for (int offset = 1; offset < NUM_WARPS_PER_BLOCK; offset *= 2) {
      int n = __shfl_up(num_pts, offset, NUM_WARPS_PER_BLOCK);

      if (lane_id >= offset)
        num_pts += n;
    }

    if (lane_id < NUM_WARPS_PER_BLOCK)
      s_num_pts[warp_id][lane_id] = num_pts;
  }

  __syncthreads();

  // Compute global offsets.
  if (warp_id == 0) {
    int sum = s_num_pts[0][NUM_WARPS_PER_BLOCK - 1];

    for (int row = 1; row < 8; ++row) {
      int tmp = s_num_pts[row][NUM_WARPS_PER_BLOCK - 1];

      if (lane_id < NUM_WARPS_PER_BLOCK)
        s_num_pts[row][lane_id] += sum;

      sum += tmp;
    }
  }

  __syncthreads();

  // Make the scan exclusive.
  if (threadIdx.x < 4 * NUM_WARPS_PER_BLOCK) {
    int val = threadIdx.x == 0 ? 0 : smem[threadIdx.x - 1];
    val += node.points_begin();
    smem[threadIdx.x] = val;
  }

  __syncthreads();

  //
  // 4- Move points.
  //

  // Output points.
  Particles &out_points = points[(params.point_selector + 1) % 2];

  // Reorder points.
  for (int range_it = range_begin + lane_id; __any(range_it < range_end);
       range_it += warpSize) {
    // Is it still an active thread?
    bool is_active = range_it < range_end;

    // Load the coordinates of the point.
    float3 p = is_active ? in_points.get_location(range_it)
                         : make_float3(0.0f, 0.0f, 0.0f);

    // Count top-left points in front.
    bool pred =
        is_active && p.x < center.x && p.y >= center.y && p.z >= center.z;
    int vote = __ballot(pred);
    int dest = s_num_pts[0][warp_id] + __popc(vote & lane_mask_lt);

    if (pred)
      out_points.set_location(dest, p);

    if (lane_id == 0)
      s_num_pts[0][warp_id] += __popc(vote);

    // Count top-right points in front.
    pred = is_active && p.x >= center.x && p.y >= center.y && p.z >= center.z;
    vote = __ballot(pred);
    dest = s_num_pts[1][warp_id] + __popc(vote & lane_mask_lt);

    if (pred)
      out_points.set_location(dest, p);

    if (lane_id == 0)
      s_num_pts[1][warp_id] += __popc(vote);

    // Count bottom-left points in front.
    pred = is_active && p.x < center.x && p.y < center.y && p.z >= center.z;
    vote = __ballot(pred);
    dest = s_num_pts[2][warp_id] + __popc(vote & lane_mask_lt);

    if (pred)
      out_points.set_location(dest, p);

    if (lane_id == 0)
      s_num_pts[2][warp_id] += __popc(vote);

    // Count bottom-right points in front.
    pred = is_active && p.x >= center.x && p.y < center.y && p.z >= center.z;
    vote = __ballot(pred);
    dest = s_num_pts[3][warp_id] + __popc(vote & lane_mask_lt);

    if (pred)
      out_points.set_location(dest, p);

    if (lane_id == 0)
      s_num_pts[3][warp_id] += __popc(vote);

    // Count top-left points in back.
    pred =
        is_active && p.x < center.x && p.y >= center.y && p.z < center.z;
    vote = __ballot(pred);
    dest = s_num_pts[4][warp_id] + __popc(vote & lane_mask_lt);

    if (pred)
      out_points.set_location(dest, p);

    if (lane_id == 0)
      s_num_pts[4][warp_id] += __popc(vote);

    // Count top-right points in back.
    pred = is_active && p.x >= center.x && p.y >= center.y && p.z < center.z;
    vote = __ballot(pred);
    dest = s_num_pts[5][warp_id] + __popc(vote & lane_mask_lt);

    if (pred)
      out_points.set_location(dest, p);

    if (lane_id == 0)
      s_num_pts[5][warp_id] += __popc(vote);

    // Count bottom-left points in back.
    pred = is_active && p.x < center.x && p.y < center.y && p.z < center.z;
    vote = __ballot(pred);
    dest = s_num_pts[6][warp_id] + __popc(vote & lane_mask_lt);

    if (pred)
      out_points.set_location(dest, p);

    if (lane_id == 0)
      s_num_pts[6][warp_id] += __popc(vote);

    // Count bottom-right points in back.
    pred = is_active && p.x >= center.x && p.y < center.y && p.z < center.z;
    vote = __ballot(pred);
    dest = s_num_pts[7][warp_id] + __popc(vote & lane_mask_lt);

    if (pred)
      out_points.set_location(dest, p);

    if (lane_id == 0)
      s_num_pts[7][warp_id] += __popc(vote);
  }

  __syncthreads();

  //
  // 5- Launch new blocks.
  //

  // The last thread launches new blocks.
  if (threadIdx.x == NUM_THREADS_PER_BLOCK - 1) {
    // The children.
    Octree_node *children = &nodes[params.num_nodes_at_this_level];

    // The offsets of the children at their level.
    int child_offset = 8 * node.id();

    // Set IDs.
    children[child_offset + 0].set_id(8 * node.id() + 0);
    children[child_offset + 1].set_id(8 * node.id() + 8);
    children[child_offset + 2].set_id(8 * node.id() + 16);
    children[child_offset + 3].set_id(8 * node.id() + 24);
    children[child_offset + 4].set_id(8 * node.id() + 32);
    children[child_offset + 5].set_id(8 * node.id() + 40);
    children[child_offset + 6].set_id(8 * node.id() + 48);
    children[child_offset + 7].set_id(8 * node.id() + 56);

    // Points of the bounding-box.
    const float3 &p_min = bbox.get_min();
    const float3 &p_max = bbox.get_max();

    // Set the bounding boxes of the children.
    // Top-left front.
    children[child_offset + 0].set_bounding_box(p_min.x, center.y, center.z,
                                                center.x, p_max.y, p_max.z);
    // Top-right front.
    children[child_offset + 1].set_bounding_box(center.x, center.y, center.z,
                                                p_max.x, p_max.y, p_max.z);
    // Bottom-left front.
    children[child_offset + 2].set_bounding_box(p_min.x, p_min.y, center.z,
                                                center.x, center.y, p_max.z);
    // Bottom-right front.
    children[child_offset + 3].set_bounding_box(center.x, p_min.y, center.z,
                                                p_max.x, center.y, p_max.z);
    // Top-left back.
    children[child_offset + 4].set_bounding_box(p_min.x, center.y, p_min.z,
                                                center.x, p_max.y, center.z);
    // Top-right back.
    children[child_offset + 5].set_bounding_box(center.x, center.y, p_min.z,
                                                p_max.x, p_max.y, center.z);
    // Bottom-left back.
    children[child_offset + 6].set_bounding_box(p_min.x, p_min.y, p_min.z,
                                                center.x, center.y, center.z);
    // Bottom-right back.
    children[child_offset + 7].set_bounding_box(center.x, p_min.y, p_min.z,
                                                p_max.x, center.y, center.z);

    // Set the ranges of the children.
    children[child_offset + 0].set_range(node.points_begin(),
                                         s_num_pts[0][warp_id]);
    children[child_offset + 1].set_range(s_num_pts[0][warp_id],
                                         s_num_pts[1][warp_id]);
    children[child_offset + 2].set_range(s_num_pts[1][warp_id],
                                         s_num_pts[2][warp_id]);
    children[child_offset + 3].set_range(s_num_pts[2][warp_id],
                                         s_num_pts[3][warp_id]);
    children[child_offset + 4].set_range(s_num_pts[3][warp_id],
                                         s_num_pts[4][warp_id]);
    children[child_offset + 5].set_range(s_num_pts[4][warp_id],
                                         s_num_pts[5][warp_id]);
    children[child_offset + 6].set_range(s_num_pts[5][warp_id],
                                         s_num_pts[6][warp_id]);
    children[child_offset + 7].set_range(s_num_pts[6][warp_id],
                                         s_num_pts[7][warp_id]);


    // Launch 4 children.
    build_octree_kernel<NUM_THREADS_PER_BLOCK>
        <<<8, NUM_THREADS_PER_BLOCK, 8 * NUM_WARPS_PER_BLOCK * sizeof(float)>>>(
            children, points, Parameters(params, true));
  }
}
