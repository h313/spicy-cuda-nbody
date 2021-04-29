#include <iostream>
#include <fstream>
#include <boost/qvm/vec.hpp>

#include "generator.h"

class DataOutput {
private:
    std::ofstream output_stream;
public:
    DataOutput(std::string &filename);
    ~DataOutput();
    void write_bounding_box(BoundingBox *bb);
    void write_datapoint(boost::qvm::vec<float, 3> pos);
    void add_datapoints(Particles &particles);
    void add_octree_bounding_boxes(OctreeNode &parent);
};
