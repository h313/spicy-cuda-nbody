/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <thrust/random.h>
#include <thrust/device_vector.h>
#include "helper_cuda.h"

////////////////////////////////////////////////////////////////////////////////
// A structure of 3D points (structure of arrays).
////////////////////////////////////////////////////////////////////////////////
class Points
{
        float *m_x;
        float *m_y;
        float *m_z;

    public:
        // Constructor.
        __host__ __device__ Points() : m_x(NULL), m_y(NULL), m_z(NULL) {}

        // Constructor.
        __host__ __device__ Points(float *x, float *y, float *z) : m_x(x), m_y(y), m_z(z) {}

        // Get a point.
        __host__ __device__ __forceinline__ float3 get_point(int idx) const
        {
            return make_float3(m_x[idx], m_y[idx], m_z[idx]);
        }

        // Set a point.
        __host__ __device__ __forceinline__ void set_point(int idx, const float3 &p)
        {
            m_x[idx] = p.x;
            m_y[idx] = p.y;
            m_y[idx] = p.z;
        }

        // Set the pointers.
        __host__ __device__ __forceinline__ void set(float *x, float *y)
        {
            m_x = x;
            m_y = y;
            m_z = z;
        }
};

////////////////////////////////////////////////////////////////////////////////
// A 3D bounding box
////////////////////////////////////////////////////////////////////////////////
class Bounding_box {
private:
  float3 m_p_min;
  float3 m_p_max;

public:
  __host__ __device__ Bounding_box() {
    m_p_min = make_float3(0.0f, 0.0f, 0.0f);
    m_p_max = make_float3(1.0f, 1.0f, 1.0f);
  }

  __host__ __device__ void compute_center(float3 &center) const {
    center.x = 0.5f * (m_p_min.x + m_p_max.x);
    center.y = 0.5f * (m_p_min.y + m_p_max.y);
    center.z = 0.5f * (m_p_min.z + m_p_max.z);
  }

  __host__ __device__ __forceinline__ const float3 &get_max() const {
    return m_p_max;
  }

  __host__ __device__ __forceinline__ const float3 &get_min() const {
    return m_p_min;
  }

  __host__ __device__ bool contains(const float3 &p) const {
    return p.x >= m_p_min.x && p.y < m_p_max.x && p.y >= m_p_min.y &&
           p.y < m_p_max.y && p.z < m_p_max.z && p.z >= m_p_min.z;
  }

  __host__ __device__ void set(float min_x, float min_y, float min_z,
                               float max_x, float max_y, float max_z) {
    m_p_min.x = min_x;
    m_p_min.y = min_y;
    m_p_min.z = min_z;
    m_p_max.x = max_x;
    m_p_max.y = max_y;
    m_p_max.z = max_z;
  }
};

////////////////////////////////////////////////////////////////////////////////
// A node of a octree.
////////////////////////////////////////////////////////////////////////////////

// TODO: Do we need change this for 3D???
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

////////////////////////////////////////////////////////////////////////////////
// Algorithm parameters.
////////////////////////////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////////////////////////////
// Build a octree on the GPU. Use CUDA Dynamic Parallelism.
//
// The algorithm works as follows. The host (CPU) launches one block of
// NUM_THREADS_PER_BLOCK threads. That block will do the following steps:
//
// 1- Check the number of points and its depth.
//
// We impose a maximum depth to the tree and a minimum number of points per
// node. If the maximum depth is exceeded or the minimum number of points is
// reached. The threads in the block exit.
//
// Before exiting, they perform a buffer swap if it is needed. Indeed, the
// algorithm uses two buffers to permute the points and make sure they are
// properly distributed in the octree. By design we want all points to be
// in the first buffer of points at the end of the algorithm. It is the reason
// why we may have to swap the buffer before leaving (if the points are in the
// 2nd buffer).
//
// 2- Count the number of points in each child.
//
// If the depth is not too high and the number of points is sufficient, the
// block has to dispatch the points into four geometrical buckets: Its
// children. For that purpose, we compute the center of the bounding box and
// count the number of points in each octant.
//
// The set of points is divided into sections. Each section is given to a
// warp of threads (32 threads). Warps use __ballot and __popc intrinsics
// to count the points. See the Programming Guide for more information about
// those functions.
//
// 3- Scan the warps' results to know the "global" numbers.
//
// Warps work independently from each other. At the end, each warp knows the
// number of points in its section. To know the numbers for the block, the
// block has to run a scan/reduce at the block level. It's a traditional
// approach. The implementation in that sample is not as optimized as what
// could be found in fast radix sorts, for example, but it relies on the same
// idea.
//
// 4- Move points.
//
// Now that the block knows how many points go in each of its 4 children, it
// remains to dispatch the points. It is straightforward.
//
// 5- Launch new blocks.
//
// The block launches four new blocks: One per children. Each of the four blocks
// will apply the same algorithm.
////////////////////////////////////////////////////////////////////////////////
template <int NUM_THREADS_PER_BLOCK>
__global__ void build_octree_kernel(Octree_node *nodes, Points *points,
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
  // octant.
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
  const Points &in_points = points[params.point_selector];

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
  Points &out_points = points[(params.point_selector + 1) % 2];

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

////////////////////////////////////////////////////////////////////////////////
// Make sure a Octree is properly defined.
////////////////////////////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////////////////////////////
// Parallel random number generator.
////////////////////////////////////////////////////////////////////////////////
struct Random_generator
{

    __host__ __device__ __forceinline__ thrust::float3 operator()()
    {
        unsigned seed = (blockIdx.x*blockDim.x + threadIdx.x);
        thrust::default_random_engine rng(seed);
        thrust::random::uniform_real_distribution<float> distrib;
        return thrust::make_float3(distrib(rng), distrib(rng), distrib(rng));
    }
};

////////////////////////////////////////////////////////////////////////////////
// Main entry point.
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // Constants to control the algorithm.
    const int num_points = 1024;
    const int max_depth  = 8;
    const int min_points_per_node = 16;

    // Find/set the device.
    // The test requires an architecture SM35 or greater (CDP capable).
    int warp_size = 0;
    int cuda_device = findCudaDevice(argc, (const char **)argv);
    cudaDeviceProp deviceProps;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProps, cuda_device));
    int cdpCapable = (deviceProps.major == 3 && deviceProps.minor >= 5) || deviceProps.major >=4;

    printf("GPU device %s has compute capabilities (SM %d.%d)\n", deviceProps.name, deviceProps.major, deviceProps.minor);

    if (!cdpCapable)
    {
        std::cerr << "cdpOctTree requires SM 3.5 or higher to use CUDA Dynamic Parallelism.  Exiting...\n" << std::endl;
        exit(EXIT_SUCCESS);
    }

    warp_size = deviceProps.warpSize;

    // Allocate memory for points.
    thrust::device_vector<float> x_d0(num_points);
    thrust::device_vector<float> x_d1(num_points);
    thrust::device_vector<float> y_d0(num_points);
    thrust::device_vector<float> y_d1(num_points);
    thrust::device_vector<float> z_d0(num_points);
    thrust::device_vector<float> z_d1(num_points);

    // Generate random points.
    Random_generator rnd;
    thrust::generate(
        thrust::make_zip_iterator(thrust::make_float3(x_d0.begin(), y_d0.begin(), z_d0.begin())),
        thrust::make_zip_iterator(thrust::make_float3(x_d0.end(), y_d0.end(), z_d0.begin())),
        rnd);

    // Host structures to analyze the device ones.
    Points points_init[2];
    points_init[0].set(thrust::raw_pointer_cast(&x_d0[0]), thrust::raw_pointer_cast(&y_d0[0]), thrust::raw_pointer_cast(&z_d0[0]));
    points_init[1].set(thrust::raw_pointer_cast(&x_d1[0]), thrust::raw_pointer_cast(&y_d1[0]), thrust::raw_pointer_cast(&z_d1[0]));

    // Allocate memory to store points.
    Points *points;
    checkCudaErrors(cudaMalloc((void **) &points, 2*sizeof(Points)));
    checkCudaErrors(cudaMemcpy(points, points_init, 2*sizeof(Points), cudaMemcpyHostToDevice));

    // We could use a close form...
    int max_nodes = 0;

    for (int i = 0, num_nodes_at_level = 1 ; i < max_depth ; ++i, num_nodes_at_level *= 8)
        max_nodes += num_nodes_at_level;

    // Allocate memory to store the tree.
    Octree_node root;
    root.set_range(0, num_points);
    Octree_node *nodes;
    checkCudaErrors(cudaMalloc((void **) &nodes, max_nodes*sizeof(Octree_node)));
    checkCudaErrors(cudaMemcpy(nodes, &root, sizeof(Octree_node), cudaMemcpyHostToDevice));

    // We set the recursion limit for CDP to max_depth.
    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, max_depth);

    // Build the octree.
    Parameters params(max_depth, min_points_per_node);
    std::cout << "Launching CDP kernel to build the octree" << std::endl;
    const int NUM_THREADS_PER_BLOCK = 128; // Do not use less than 128 threads.
    const int NUM_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / warp_size;
    const size_t smem_size = 8*NUM_WARPS_PER_BLOCK*sizeof(int);
    build_octree_kernel<NUM_THREADS_PER_BLOCK><<<1, NUM_THREADS_PER_BLOCK, smem_size>>>(nodes, points, params);
    checkCudaErrors(cudaGetLastError());

    // Copy points to CPU.
    thrust::host_vector<float> x_h(x_d0);
    thrust::host_vector<float> y_h(y_d0);
    thrust::host_vector<float> z_h(y_d0);
    Points host_points;
    host_points.set(thrust::raw_pointer_cast(&x_h[0]), thrust::raw_pointer_cast(&y_h[0]), hrust::raw_pointer_cast(&z_h[0]));

    // Copy nodes to CPU.
    Octree_node *host_nodes = new Octree_node[max_nodes];
    checkCudaErrors(cudaMemcpy(host_nodes, nodes, max_nodes *sizeof(Octree_node), cudaMemcpyDeviceToHost));

    // Validate the results.
    bool ok = check_octree(host_nodes, 0, num_points, &host_points, params);
    std::cout << "Results: " << (ok ? "OK" : "FAILED") << std::endl;

    // Free CPU memory.
    delete[] host_nodes;

    // Free memory.
    checkCudaErrors(cudaFree(nodes));
    checkCudaErrors(cudaFree(points));

    cudaDeviceReset();

    exit(ok ? EXIT_SUCCESS : EXIT_FAILURE);
}



