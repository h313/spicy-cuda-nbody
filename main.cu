#include "generation.h"
#include "gravity_calculations.h"
#include <iostream>

int main(int argc, char **argv) {
  const int num_points = 1024;
  const int max_depth = 8;
  const int min_points_per_node = 16;

  // Find/set the device.
  // The test requires an architecture SM35 or greater (CDP capable).
  int warp_size = 0;
  int cuda_device = findCudaDevice(argc, (const char **)argv);
  cudaDeviceProp deviceProps;
  checkCudaErrors(cudaGetDeviceProperties(&deviceProps, cuda_device));
  int cdpCapable = (deviceProps.major == 3 && deviceProps.minor >= 5) ||
                   deviceProps.major >= 4;

  printf("GPU device %s has compute capabilities (SM %d.%d)\n",
         deviceProps.name, deviceProps.major, deviceProps.minor);

  warp_size = deviceProps.warpSize;

  Particles *particles;
  Particles particles_init[2];
  particles_init[0] = *generate_random(2, 1.0f);
  particles_init[1] = *generate_random(2, 1.0f);

  for (int i = 0; i < 2; i++) {
    float3 item = particles_init[0].get_location(i);
    std::cout << "Particle " << i << ": <" << item.x << ", " << item.y << ", "
              << item.z << ">" << std::endl;
  }

  checkCudaErrors(cudaMalloc((void **)&particles, 2 * sizeof(Particles)));
  checkCudaErrors(cudaMemcpy(particles, particles_init, 2 * sizeof(Particles),
                             cudaMemcpyHostToDevice));

  int max_nodes = 0;

  for (int i = 0, num_nodes_at_level = 1; i < max_depth;
       ++i, num_nodes_at_level *= 4)
    max_nodes += num_nodes_at_level;

  // Allocate memory to store the tree.
  Octree_node root;
  root.set_range(0, num_points);
  Octree_node *nodes;
  checkCudaErrors(
      cudaMalloc((void **)&nodes, max_nodes * sizeof(Octree_node)));
  checkCudaErrors(
      cudaMemcpy(nodes, &root, sizeof(Octree_node), cudaMemcpyHostToDevice));

  return 0;
}
